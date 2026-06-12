from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


class SVDNystromLlamaAttention(nn.Module):
    """Minimal SVD-Nystrom replacement for one Hugging Face LLaMA attention layer.

    Usage pattern:
      1. Patch one or more layers with `patch_llama_attention`.
      2. Turn on prefill mode and run the prompt once.
      3. Turn off prefill mode and feed new tokens autoregressively.

    The prompt keys/values and SVD descriptors are fixed after prefill. New
    tokens get exact local attention plus a global SVD-Nystrom path:

        global(Q) = softmax(Q B) @ solve(M, R @ V_prompt)

    where B comes from the prompt K SVD, A is either B or the prompt Q SVD, and

        M = softmax(A^T B)
        R = softmax(A^T K_prompt^T)
    """

    def __init__(self, old_attn: nn.Module, rank: int = 64, local_window: int = 1024, eps: float = 1e-4, lamba: float = 0.5, svd_mode: str = "svd_k") -> None:
        super().__init__()

        self.config = old_attn.config
        self.layer_idx = old_attn.layer_idx
        self.head_dim = old_attn.head_dim
        self.num_key_value_groups = old_attn.num_key_value_groups
        self.attention_dropout = old_attn.attention_dropout
        self.scaling = getattr(old_attn, "scaling", self.head_dim**-0.5)
        self.q_proj = old_attn.q_proj
        self.k_proj = old_attn.k_proj
        self.v_proj = old_attn.v_proj
        self.o_proj = old_attn.o_proj

        # Self chosen parameters 
        self.rank = rank
        self.local_window = local_window
        self.eps = eps
        self.lamba = lamba
        self.svd_mode = svd_mode
        self.prefill = False
        self.clear()

    def clear(self) -> None:
        # Clear cache
        self.prefix_len = 0
        self.generated_len = 0
        self.prefix_k = None
        self.prefix_v = None
        self.target_k = None
        self.target_v = None
        self.V_K = None
        self.const_cache = None

    def project(self, hidden_states, position_embeddings):
        # Takes hidden_states and turns them into the actual attention tensors Q, K, and V used inside the attention layer.
        shape = hidden_states.shape[:-1]
        q = self.q_proj(hidden_states).view(*shape, -1, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(*shape, -1, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(*shape, -1, self.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)
        return shape, q, k, v

    def dense_causal_attention(self, q, k, v, attention_mask):
        # Calculates full attention in the normal llama way
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        if attention_mask is None:
            t = q.shape[-2]
            causal = torch.ones(t, t, device=q.device, dtype=torch.bool).tril()
            scores = scores.masked_fill(~causal.view(1, 1, t, t), float("-inf"))
        else:
            scores = scores + attention_mask
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        return torch.matmul(weights, v)

    def top_right_singular_vectors(self, x):
        """A rank-limited feature-space basis for the input tokens.

        Input has shape [batch, heads, tokens, head_dim]. The completed function
        produces landmark directions with shape [batch, heads, head_dim, rank],
        suitable for dot products with query/key vectors.
        """
        _, _, vh = torch.linalg.svd(x.float(), full_matrices=False)
        rank = min(self.rank, vh.shape[-2])
        return vh[..., :rank, :].transpose(-2, -1).contiguous()

    def build_prompt_cache(self, q, k, v):
        """Prompt-only state used by the global path during later tokens.

        The completed function records the prompt length, prompt K/V tensors,
        and compressed global descriptors. After it runs, `self.B` contains the
        key-side landmark directions and `self.const_cache` contains the value table that
        `global_attention` reads from. Later tokens do not need to recompute
        prompt descriptors or prompt value summaries.
        """
        q = q.detach()
        k = k.detach()
        v = v.detach()
        self.prefix_len = k.shape[-2]
        self.prefix_k = k
        self.prefix_v = v
        self.target_k = None
        self.target_v = None
        self.generated_len = 0

        V_K = self.top_right_singular_vectors(k).transpose(-2, -1)
        V_Q = self.top_right_singular_vectors(q).transpose(-2, -1)

        M = F.softmax(V_Q @ V_K.transpose(-2, -1) * self.scaling, dim=-1)      # [r, r]
        R = F.softmax((V_Q @ k.float().transpose(-2, -1)) * self.scaling, dim=-1)    # [r, n]

        eye = torch.eye(M.shape[-1], device=M.device, dtype=M.dtype)
        M = (1.0 - self.eps) * M + self.eps * eye[None, None, :, :]

        # Saves as class variables
        self.V_K = V_K.to(k.dtype)
        self.const_cache = torch.linalg.solve(M, R @ v.float()).to(v.dtype)

    def local_attention(self, q, k, v):
        """Exact causal attention over the recent token neighborhood.

        The completed function attends from the current query tokens to cached
        prompt tokens, cached generated tokens, and the current token(s), while
        excluding future keys. If `local_window` is set, keys older than that
        lookback are excluded too. The output shape matches q.
        """
        q_len = q.shape[-2]
        q_start = self.prefix_len + self.generated_len
        current_token_positions = torch.arange(q_start, q_start + q_len, device=q.device)

        keys = [self.prefix_k]
        values = [self.prefix_v]
        key_pos = [torch.arange(self.prefix_len, device=q.device)]

        if self.target_k is not None:
            keys.append(self.target_k)
            values.append(self.target_v)
            key_pos.append(torch.arange(self.prefix_len, self.prefix_len + self.target_k.shape[-2], device=q.device))

        keys.append(k)
        values.append(v)
        key_pos.append(current_token_positions)

        K = torch.cat(keys, dim=-2)
        V = torch.cat(values, dim=-2)
        cache_token_positions = torch.cat(key_pos)

        # Bygger causal mask
        allowed = cache_token_positions.view(1, -1) <= current_token_positions.view(-1, 1)
        if self.local_window is not None:
            allowed = allowed & ((current_token_positions.view(-1, 1) - cache_token_positions.view(1, -1)) < self.local_window)

        mask = allowed.view(1, 1, q_len, K.shape[-2])
        return F.scaled_dot_product_attention(q, K, V, attn_mask=mask, dropout_p=0.0, scale=self.scaling) # llamas attention scaling: self.scaling = 1 / sqrt(head_dim)

    def global_attention(self, q):
        """Approximate long-range attention through the cached prompt.

        The completed function maps current query tokens into the cached
        landmark space and reads from the fixed prompt value table. It does not
        inspect generated-token K/V state. The output shape matches q.
        """
        weights = F.softmax(q @ self.V_K.transpose(-2, -1) * self.scaling, dim=-1, dtype=torch.float32)
        return torch.matmul(weights.to(q.dtype), self.const_cache)

    def append_target_kv(self, k, v):
        self.target_k = k if self.target_k is None else torch.cat([self.target_k, k], dim=-2)
        self.target_v = v if self.target_v is None else torch.cat([self.target_v, v], dim=-2)
        self.generated_len += k.shape[-2]

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kwargs):
        """Prompt prefill or new-token attention, depending on `self.prefill`.

        In prefill mode, the completed function builds the prompt cache and
        produces ordinary causal attention for the prompt. Outside prefill mode,
        it produces new-token attention from exact local context plus compressed
        global prompt context, then appends the new K/V tensors to the generated
        cache.
        """
        del past_key_values, kwargs
        shape, q, k, v = self.project(hidden_states, position_embeddings)

        if self.prefill:
            self.build_prompt_cache(q, k, v)
            out = self.dense_causal_attention(q, k, v, attention_mask)
            
        else:
            local_out = self.local_attention(q, k, v)
            global_out = self.global_attention(q)

            if self.layer_idx == 5:
                print("\n--- ATTENTION MIX DEBUG ---")
                print(f"layer_idx: {self.layer_idx}")
                print(f"lambda: {self.lamba}")
                print(f"local_out norm: {local_out.float().norm().item():.4f}")
                print(f"global_out norm: {global_out.float().norm().item():.4f}")
                print(f"global/local ratio: {global_out.float().norm().item() / (local_out.float().norm().item() + 1e-8)}")

            out = self.lamba * global_out + (1 - self.lamba) * local_out
            self.append_target_kv(k, v)

        out = out.transpose(1, 2).reshape(*shape, -1).contiguous()
        return self.o_proj(out), None


def patch_llama_attention(
    model,
    layers=2,
    rank: int = 64,
    local_window: int = 1024,
    eps: float = 1e-4,
    lamba: float = 1.0,
    svd_mode: str = "svd_k"):

    if layers == "all":
        layers = range(len(model.model.layers))
    elif isinstance(layers, int):
        layers = [layers]
    elif isinstance(layers, str):
        layers = [int(x) for x in layers.split(",")]

    for i in layers:
        model.model.layers[i].self_attn = SVDNystromLlamaAttention(
            model.model.layers[i].self_attn,
            rank=rank,
            local_window=local_window,
            eps=eps,
            lamba=lamba,
            svd_mode=svd_mode)
        
    return model


def set_prefill(model, enabled: bool):
    for module in model.modules():
        if isinstance(module, SVDNystromLlamaAttention):
            module.prefill = enabled


def clear_nystrom(model):
    for module in model.modules():
        if isinstance(module, SVDNystromLlamaAttention):
            module.clear()
