import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from fagprojekt.SVD import decompose_K

"""
Her prøver vi at erstatte HuggingFace's normale KV-cache for ét LlamaAttention-layer.

I stedet for at gemme hele promptens K-cache, gemmer vi K som SVD-faktorer.
V gemmes stadig normalt, og nye K-tokens under generation gemmes bare exact.

Det er mest en prototype, så vi kan teste hvor SVD eller Hokus Pokus kan sættes
ind i modellen.
"""


def svd_cache_attention(
    module: nn.Module,
    query_states: torch.Tensor,
    value_cache: torch.Tensor,
    key_tail_cache: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    dropout: float) -> torch.Tensor:
    """
    Her beregner vi attention med vores egen SVD-cache.

    Promptens K ligger ikke som full K længere, men som SVD-faktorer A og B.
    De nye K-tokens, der kommer under generation, ligger stadig exact i en lille
    tail-cache.

    Det er her modellen faktisk bruger den komprimerede K-cache under decoding.
    """

    batch_size, num_query_heads, q_len, _ = query_states.shape

    assert batch_size == 1
    assert q_len == 1

    outputs = []

    for query_head_idx in range(num_query_heads):
        kv_head_idx = query_head_idx // module.num_key_value_groups

        Q = query_states[0, query_head_idx, :, :].float()

        A = module.svd_key_cache[kv_head_idx]["A"].float()
        B = module.svd_key_cache[kv_head_idx]["B"].float()

        prompt_len = A.shape[0]

        V_full = value_cache[0, kv_head_idx, :, :].float()
        V_prompt = V_full[:prompt_len, :]

        prompt_scores = (Q @ B @ A.T) * module.scaling

        if key_tail_cache is not None:
            K_tail = key_tail_cache[0, kv_head_idx, :, :].float()
            V_tail = V_full[prompt_len:, :]

            tail_scores = (Q @ K_tail.T) * module.scaling

            scores = torch.cat([prompt_scores, tail_scores], dim=-1)
            values = torch.cat([V_prompt, V_tail], dim=0)
        else:
            scores = prompt_scores
            values = V_prompt

        if attention_mask is not None:
            scores = scores + attention_mask[0, 0, :, -scores.shape[-1]:].float()

        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        weights = F.dropout(weights, p=dropout, training=module.training)

        head_output = weights @ values.to(weights.dtype)
        outputs.append(head_output)

    return torch.stack(outputs, dim=0).unsqueeze(0)


class SVDLlamaAttentionWrapper(nn.Module):
    """
    Wrapper for ét LlamaAttention-layer.

    Den bruger ikke HuggingFace's KV-cache for dette layer.
    I stedet gemmes prompt-K som SVD, prompt-V fuldt,
    og generated-token K gemmes exact som en lille tail.
    """

    def __init__(self, original_attn, rank: int = 16):
        super().__init__()

        self.layer_idx = original_attn.layer_idx
        self.head_dim = original_attn.head_dim
        self.num_key_value_groups = original_attn.num_key_value_groups
        self.attention_dropout = original_attn.attention_dropout
        self.scaling = original_attn.scaling

        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.o_proj = original_attn.o_proj

        self.rank = rank

        self.svd_key_cache = None
        self.value_cache = None
        self.key_tail_cache = None

    def reset_cache(self):
        """
        Resetter vores egne caches så gamle K/V-værdier fra en tidligere prompt ikke fucker med de nye.
        """
        self.svd_key_cache = None
        self.value_cache = None
        self.key_tail_cache = None

    def build_svd_key_cache(self, key_states: torch.Tensor):
        """
        Laver SVD-cache af promptens K.

        Vi går igennem hver KV-head, laver SVD af dens K-matrix og gemmer kun A og B.
        Full prompt-K bliver ikke gemt bagefter.
        """

        assert key_states.shape[0] == 1

        self.svd_key_cache = {}

        num_kv_heads = key_states.shape[1]

        for kv_head_idx in range(num_kv_heads):
            K_head = key_states[0, kv_head_idx, :, :].float()
            A, B = decompose_K(K_head, self.rank)

            self.svd_key_cache[kv_head_idx] = {"A": A, "B": B}

    def normal_prefill_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float) -> torch.Tensor:
        """
        Normal attention, næsten som HuggingFace gør det.

        Vi bruger den under prefill, fordi prompten stadig kommer ind samlet her.
        SVD-cachen bygges under prefill, men selve prefill-attention beregnes stadig
        normalt.
        """

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        scores = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            scores = scores + attention_mask

        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        weights = F.dropout(weights, p=dropout, training=self.training)

        return torch.matmul(weights, value_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        cache_position: torch.LongTensor | None = None,
        **kwargs) -> tuple[torch.Tensor, None]:
        """
        Kører attention-layeret.

        Først laver vi Q, K og V som i normal LlamaAttention. Under prefill bygger vi
        SVD-cachen for prompt-K og gemmer V. Under decoding bruger vi SVD-K for prompten
        og exact K-tail for de nye tokens.
        """

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        q_len = query_states.shape[2]
        dropout = 0.0 if not self.training else self.attention_dropout

        if q_len > 1:
            # Prefill: lav SVD-cache af prompt-K og gem prompt-V.
            self.build_svd_key_cache(key_states)
            self.value_cache = value_states.clone()
            self.key_tail_cache = None

            key_len = key_states.shape[-2]
            if attention_mask is not None:
                attention_mask = attention_mask[..., -key_len:]

            attn_output = self.normal_prefill_attention(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attention_mask=attention_mask,
                dropout=dropout)

        else:
            # Decoding: gem ny K i tail og append ny V.
            if self.key_tail_cache is None:
                self.key_tail_cache = key_states.clone()
            else:
                self.key_tail_cache = torch.cat(
                    [self.key_tail_cache, key_states],
                    dim=2)

            if self.value_cache is None:
                self.value_cache = value_states.clone()
            else:
                self.value_cache = torch.cat(
                    [self.value_cache, value_states],
                    dim=2)

            total_key_len = self.value_cache.shape[2]
            if attention_mask is not None:
                attention_mask = attention_mask[..., -total_key_len:]

            attn_output = svd_cache_attention(
                module=self,
                query_states=query_states,
                value_cache=self.value_cache,
                key_tail_cache=self.key_tail_cache,
                attention_mask=attention_mask,
                dropout=dropout)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None