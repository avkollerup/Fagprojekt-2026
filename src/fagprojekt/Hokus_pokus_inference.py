import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


"""
Hokus Pokus KV-cache wrapper.

Prefill:
    Build compressed prompt cache:
        B     [d, r]
        cache [r, d]

Decoding:
    Use compressed prompt cache for old tokens.
    Store generated K/V exactly in a small tail cache.
"""


class HokusPokusMLP(nn.Module):
    """Optional learned g_theta. Not trained by default."""

    def __init__(self, rank: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(rank, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rank),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)


def stable_inverse(M: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Solves (M + eps I)X = I instead of using torch.inverse."""

    r = M.shape[-1]
    eye = torch.eye(r, device=M.device, dtype=M.dtype)
    return torch.linalg.solve(M + eps * eye, eye)


def build_hokus_pokus_cache_for_head(
    K_head: torch.Tensor,
    V_head: torch.Tensor,
    rank: int,
    scaling: float,
    eps: float = 1e-4,
):
    """Builds B and compressed cache for one KV-head."""

    K = K_head.float()
    V = V_head.float()

    _, _, Vt = torch.linalg.svd(K, full_matrices=False)

    r_eff = min(rank, Vt.shape[0])
    B = Vt[:r_eff].T.contiguous()

    A = B

    R = F.softmax((A.T @ K.T) * scaling, dim=-1)
    S = R @ V

    M = F.softmax((A.T @ B) * scaling, dim=-1)
    cache = stable_inverse(M, eps=eps) @ S

    return B.to(K_head.dtype), cache.to(K_head.dtype)


def hokus_pokus_decode_attention(
    module: nn.Module,
    query_states: torch.Tensor,
    key_tail_cache: torch.Tensor | None,
    value_tail_cache: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    dropout: float,
) -> torch.Tensor:
    """Decoding with compressed prompt cache + exact generated-token tail."""

    batch_size, num_query_heads, q_len, _ = query_states.shape

    assert batch_size == 1
    assert q_len == 1

    outputs = []

    for query_head_idx in range(num_query_heads):
        kv_head_idx = query_head_idx // module.num_key_value_groups

        q_t = query_states[0, query_head_idx, :, :].float()

        B = module.hp_prompt_cache[kv_head_idx]["B"].float()
        cache = module.hp_prompt_cache[kv_head_idx]["cache"].float()

        qB = q_t @ B

        if module.g_theta is None:
            L_t = F.softmax(qB * module.scaling, dim=-1)
        else:
            L_t = module.g_theta(qB.to(query_states.dtype)).float()

        global_out = L_t @ cache

        if key_tail_cache is None or value_tail_cache is None:
            local_out = torch.zeros_like(global_out)
        else:
            K_tail = key_tail_cache[0, kv_head_idx, :, :].float()
            V_tail = value_tail_cache[0, kv_head_idx, :, :].float()

            scores = (q_t @ K_tail.T) * module.scaling

            if attention_mask is not None:
                scores = scores + attention_mask[0, 0, :, -scores.shape[-1]:].float()

            weights = F.softmax(scores, dim=-1, dtype=torch.float32)
            weights = F.dropout(weights, p=dropout, training=module.training)

            local_out = weights @ V_tail

        outputs.append((global_out + local_out).to(query_states.dtype))

    return torch.stack(outputs, dim=0).unsqueeze(0)


class HokusPokusLlamaAttentionWrapper(nn.Module):
    """Wrapper for one LlamaAttention layer."""

    def __init__(
        self,
        original_attn: nn.Module,
        rank: int = 16,
        g_theta: nn.Module | None = None,
        inverse_eps: float = 1e-4,
    ):
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
        self.g_theta = g_theta
        self.inverse_eps = inverse_eps

        self.hp_prompt_cache = None
        self.key_tail_cache = None
        self.value_tail_cache = None

    def reset_cache(self):
        self.hp_prompt_cache = None
        self.key_tail_cache = None
        self.value_tail_cache = None

    def build_hokus_pokus_prompt_cache(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ):
        """Builds compressed prompt cache for all KV-heads."""

        self.hp_prompt_cache = {}

        for kv_head_idx in range(key_states.shape[1]):
            B, cache = build_hokus_pokus_cache_for_head(
                K_head=key_states[0, kv_head_idx],
                V_head=value_states[0, kv_head_idx],
                rank=self.rank,
                scaling=self.scaling,
                eps=self.inverse_eps,
            )

            self.hp_prompt_cache[kv_head_idx] = {
                "B": B,
                "cache": cache,
            }

    def normal_prefill_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float,
    ) -> torch.Tensor:
        """Normal attention during prefill."""

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        scores = query_states @ key_states.transpose(2, 3)
        scores = scores * self.scaling

        if attention_mask is not None:
            scores = scores + attention_mask[:, :, :, : key_states.shape[-2]]

        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        weights = F.dropout(weights, p=dropout, training=self.training)

        return weights @ value_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if position_embeddings is None:
            raise ValueError("position_embeddings is needed for RoPE.")

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
        )

        q_len = query_states.shape[2]
        dropout = 0.0 if not self.training else self.attention_dropout

        if q_len > 1:
            self.build_hokus_pokus_prompt_cache(key_states, value_states)

            self.key_tail_cache = None
            self.value_tail_cache = None

            attn_output = self.normal_prefill_attention(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attention_mask=attention_mask,
                dropout=dropout,
            )

        else:
            if self.hp_prompt_cache is None:
                raise RuntimeError("Run prefill before decoding.")

            if self.key_tail_cache is None:
                self.key_tail_cache = key_states.clone()
                self.value_tail_cache = value_states.clone()
            else:
                self.key_tail_cache = torch.cat([self.key_tail_cache, key_states], dim=2)
                self.value_tail_cache = torch.cat([self.value_tail_cache, value_states], dim=2)

            if attention_mask is not None:
                tail_len = self.key_tail_cache.shape[2]
                attention_mask = attention_mask[..., -tail_len:]

            attn_output = hokus_pokus_decode_attention(
                module=self,
                query_states=query_states,
                key_tail_cache=self.key_tail_cache,
                value_tail_cache=self.value_tail_cache,
                attention_mask=attention_mask,
                dropout=dropout,
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None