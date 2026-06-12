import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


"""
Hokus Pokus KV-cache wrapper.

Based on two frameworks from supervisor documents:

1. "Prefix Landmark Attention via Global Compression (SVD)"
   - SVD of prompt K gives B (frozen landmark basis)
   - L = softmax(QB) maps tokens to landmark space
   - Causality is free: B is frozen from past-only prompt
   - Local path handles generated-token causality

2. "Causal Landmark Attention via Average Pooling"
   - R replaced by fixed average pooling over uniform segments
   - Hybrid global (Nystrom) + local (exact tail) architecture

Hokus Pokus contribution (not in supervisor documents):
   - g_theta(QB) replaces softmax(QB) * M^{-1}
   - A learned MLP absorbs the Nystrom M^{-1} correction implicitly,
     avoiding the explicit and numerically unstable matrix inversion.

Prefill:
    Build compressed prompt cache:
        B     [d, r]   - right singular vectors of K (frozen landmark basis)
        cache [r, d]   - S = R_pool @ V  (M^{-1} omitted, absorbed by g_theta)

Decoding:
    Global path:  g_theta(q_t B) @ cache           attend to compressed prompt
    Local path:   softmax(q_t K_tail^T) @ V_tail   attend to generated tokens
    Output:       global + local
"""


def build_average_pool(n: int, r: int, device: torch.device) -> torch.Tensor:
    """
    Builds fixed average pooling matrix R_pool of shape [r, n].

    From "Causal Landmark Attention via Average Pooling":
        "An elegant and highly efficient alternative is to simply use Average
        Pooling over uniform segments. In this case, we replace the learned
        softmax R with a fixed fraction matrix Rpool."

    Each row k covers tokens in segment k, with values 1/segment_size,
    so that R_pool @ V gives the mean of V vectors in each segment.
    """
    R = torch.zeros(r, n, device=device)
    for k in range(r):
        start = min(int(k * n / r), n - 1)
        end   = min(int((k + 1) * n / r), n)
        end   = max(end, start + 1)
        R[k, start:end] = 1.0 / (end - start)
    return R


def stable_inverse(M: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Solves (M + eps I) X = I for X, i.e. computes a regularised inverse.
    More stable than torch.inverse for near-singular M.

    Kept for reference — not used in current implementation since
    M^{-1} is absorbed by g_theta instead.
    """
    r = M.shape[-1]
    eye = torch.eye(r, device=M.device, dtype=M.dtype)
    return torch.linalg.solve(M + eps * eye, eye)


def build_hokus_pokus_cache_for_head(
    K_head: torch.Tensor,   # [n, d]
    V_head: torch.Tensor,   # [n, d]
    rank: int,
    scaling: float,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Builds B and compressed cache for one KV-head.

    Steps 1-2 from "Prefix Landmark Attention via Global Compression (SVD)":
        "Run Truncated SVD: K ≈ UΣVᵀ. The right singular vectors V ∈ R^{d×r}
        represent the top r principal directions in the embedding space.
        We set both our Key Landmarks and Query Landmarks to this basis: B = V."

    Steps 3-4 from "Causal Landmark Attention via Average Pooling":
        R_pool replaces the learned softmax(A^T K^T) with a fixed average
        over uniform token segments. cache = R_pool @ V gives the mean
        of V vectors per segment, used as compressed prompt memory.

    Hokus Pokus modification:
        M^{-1} correction is intentionally omitted from cache. Computing
        M = softmax(B^T B) produces a near-uniform [r, r] matrix whose
        inverse amplifies cache norms by a factor of ~r, corrupting the
        residual stream. Instead cache = S directly, and g_theta is trained
        to absorb the M^{-1} correction implicitly.

    Steps:
        1. SVD of K -> U [n,r], S_vals [r], Vt [r,d]
        2. B = V_r^T          [d, r]   frozen landmark basis
        3. R_pool             [r, n]   average pooling over prompt segments
        4. cache              [r, d]   S = R_pool @ V
    """
    K = K_head.float()   # [n, d]
    V = V_head.float()   # [n, d]
    n, d = K.shape

    # Step 1 — SVD of K
    # From supervisor doc: "Run Truncated SVD: K ≈ UΣVᵀ"
    # U:      [n, min(n,d)]
    # S_vals: [min(n,d)]
    # Vt:     [min(n,d), d]
    U, S_vals, Vt = torch.linalg.svd(K, full_matrices=False)
    r_eff = min(rank, S_vals.shape[0])

    assert n >= r_eff, (
        f"Prompt length {n} must be >= rank {r_eff}. "
        f"Use a shorter rank or a longer prompt."
    )

    # Step 2 — B: top-r right singular vectors of K, shape [d, r]
    # From supervisor doc: "The right singular vectors V ∈ R^{d×r} represent
    # the top r principal directions. We set B = V."
    # B is frozen after prefill and never updated during decoding.
    B = Vt[:r_eff].T.contiguous()                      # [d, r]

    # Step 3 — Average pooling matrix, shape [r, n]
    # From supervisor doc: "use Average Pooling over uniform segments...
    # replace the learned softmax R with a fixed fraction matrix Rpool."
    # Divides the n prompt tokens into r uniform segments.
    R_pool = build_average_pool(n, r_eff, K.device)    # [r, n]

    # Step 4 — Compressed cache, shape [r, d]
    # cache[k] is the average of V vectors in segment k (= S in Nystrom notation).
    # M^{-1} omitted — absorbed by g_theta during training.
    # At decoding time: output = g_theta(q_t B) @ cache
    cache = R_pool @ V                                  # [r, d]

    return B.to(K_head.dtype), cache.to(K_head.dtype)


class HokusPokusMLP(nn.Module):
    """
    Learnable g_theta — the core Hokus Pokus contribution.

    Replaces softmax(QB) * M^{-1} from the Nystrom framework with a single
    learned MLP. This avoids computing M^{-1} explicitly, which is numerically
    unstable when M = softmax(B^T B) is near-uniform.

    Architecture:
        Linear -> ReLU -> Linear -> Softmax -> Linear

    The final nn.Linear after the softmax approximates the M^{-1} correction
    that the Nystrom framework requires explicitly. The softmax ensures the
    intermediate representation is a valid attention distribution before
    the correction is applied.

    Input:  q_t B   [*, r]   query projected into landmark space
    Output: L_t     [*, r]   approximate attention weights over landmarks
    """

    def __init__(self, rank: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(rank, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rank),
            nn.Softmax(dim=-1),
            nn.Linear(rank, rank),   # approximates M^{-1} correction
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def hokus_pokus_decode_attention(
    module: nn.Module,
    query_states: torch.Tensor,            # [1, num_query_heads, 1, head_dim]
    key_tail_cache: torch.Tensor | None,   # [1, num_kv_heads, t, head_dim]
    value_tail_cache: torch.Tensor | None, # [1, num_kv_heads, t, head_dim]
    attention_mask: torch.Tensor | None,   # already sliced to tail length
    dropout: float,
) -> torch.Tensor:
    """
    Decoding attention combining the two paths from the supervisor documents.

    Global path — from "Prefix Landmark Attention via Global Compression (SVD)":
        "As the model autoregressively generates new tokens, they simply use
        their new Queries (Qnew) to evaluate L = softmax(QnewB). Because B, A,
        and K are frozen representations of the past prompt, no future
        information can leak."
        Hokus Pokus replaces softmax(QB) with g_theta(QB).

        g_theta(q_t B) @ cache      [1, d]   O(r*d) per step

    Local path — from "Causal Landmark Attention via Average Pooling":
        "The Local Path (Standard Attention): A short sliding window handles
        exact, high-detail local context. This provides the missing
        intra-segment causality."

        softmax(q_t K_tail^T) @ V_tail  [1, d]   O(t*d) per step

    Output: global + local          [1, d]
    """
    batch_size, num_query_heads, q_len, _ = query_states.shape

    assert batch_size == 1
    assert q_len == 1

    outputs = []

    for query_head_idx in range(num_query_heads):
        kv_head_idx = query_head_idx // module.num_key_value_groups

        # q_t: [1, head_dim]
        q_t = query_states[0, query_head_idx, 0:1, :].float()

        B     = module.hp_prompt_cache[kv_head_idx]["B"].float()      # [d, r]
        cache = module.hp_prompt_cache[kv_head_idx]["cache"].float()  # [r, d]

        # --- GLOBAL PATH ---
        # From supervisor doc: "L = softmax(QB)" — tokens reading landmarks.
        # Hokus Pokus: replace softmax(QB) with g_theta(QB).
        # Causality is free: B is frozen from past-only prompt.
        qB = q_t @ B    # [1, r]  project query into landmark space

        if module.g_theta is None:
            # Baseline (no trained g): plain softmax approximation
            # Equivalent to L from supervisor doc without M^{-1} correction
            L_t = F.softmax(qB * module.scaling, dim=-1)              # [1, r]
        else:
            # Hokus Pokus: g_theta approximates softmax(QB) * M^{-1}
            L_t = module.g_theta(qB.to(query_states.dtype)).float()   # [1, r]

        global_out = L_t @ cache   # [1, d]

        # --- LOCAL PATH ---
        # From supervisor doc: "standard lower-triangular causal attention
        # just for the newly generated tokens."
        if key_tail_cache is None or value_tail_cache is None:
            # First decoding step — no generated tokens yet
            local_out = torch.zeros_like(global_out)
        else:
            K_tail = key_tail_cache[0, kv_head_idx, :, :].float()    # [t, d]
            V_tail = value_tail_cache[0, kv_head_idx, :, :].float()  # [t, d]

            scores = (q_t @ K_tail.T) * module.scaling   # [1, t]

            # attention_mask already sliced to tail length in forward()
            if attention_mask is not None:
                scores = scores + attention_mask[0, 0, :, :].float()

            weights = F.softmax(scores, dim=-1, dtype=torch.float32)
            weights = F.dropout(weights, p=dropout, training=module.training)

            local_out = weights @ V_tail   # [1, d]

        # From supervisor doc: "combining the local sliding window with our
        # causally masked Nystrom approximation"
        outputs.append((global_out + local_out).to(query_states.dtype))

    # Stack heads: [num_query_heads, 1, d] -> unsqueeze batch -> [1, num_query_heads, 1, d]
    return torch.stack(outputs, dim=0).unsqueeze(0)


class HokusPokusLlamaAttentionWrapper(nn.Module):
    """
    Wrapper for one LlamaAttention layer using Hokus Pokus KV-cache compression.

    Implements the hybrid architecture from the supervisor documents:

    Prefill — "Prefix Landmark Attention via Global Compression (SVD)":
        Exact attention runs unchanged. Simultaneously builds the compressed
        prompt cache: B (frozen SVD basis) and cache (average-pooled V summary).

    Decoding — hybrid of both supervisor documents:
        Global path attends to the compressed prompt via g_theta(q_t B) @ cache.
        Local path attends to generated tokens via exact causal attention.
        Output = global + local.

    Causality — "Prefix Landmark Attention via Global Compression (SVD)":
        "Because B, A, and K are frozen representations of the past prompt,
        no future information can leak." Structural guarantee, not approximation.
    """

    def __init__(
        self,
        original_attn: nn.Module,
        rank: int = 64,
        g_theta: nn.Module | None = None,
        inverse_eps: float = 1e-4,
    ):
        super().__init__()

        self.layer_idx            = original_attn.layer_idx
        self.head_dim             = original_attn.head_dim
        self.num_key_value_groups = original_attn.num_key_value_groups
        self.attention_dropout    = original_attn.attention_dropout
        self.scaling              = original_attn.scaling

        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.o_proj = original_attn.o_proj

        self.rank        = rank
        self.g_theta     = g_theta
        self.inverse_eps = inverse_eps

        # Caches — populated during prefill, reset between sequences
        self.hp_prompt_cache  = None   # dict: kv_head_idx -> {B, cache}
        self.key_tail_cache   = None   # [1, num_kv_heads, t, head_dim]
        self.value_tail_cache = None   # [1, num_kv_heads, t, head_dim]

    def reset_cache(self):
        """Must be called between sequences to avoid stale cache."""
        self.hp_prompt_cache  = None
        self.key_tail_cache   = None
        self.value_tail_cache = None

    def build_hokus_pokus_prompt_cache(
        self,
        key_states: torch.Tensor,     # [1, num_kv_heads, n, head_dim]
        value_states: torch.Tensor,   # [1, num_kv_heads, n, head_dim]
    ):
        """
        Builds B and cache for every KV-head from the prompt K and V.

        Called once during prefill. After this, B and cache are frozen
        for the entire generation — new tokens cannot modify them.
        """
        assert key_states.shape[0] == 1, "Batch size must be 1"

        self.hp_prompt_cache = {}

        for kv_head_idx in range(key_states.shape[1]):
            B, cache = build_hokus_pokus_cache_for_head(
                K_head=key_states[0, kv_head_idx],    # [n, head_dim]
                V_head=value_states[0, kv_head_idx],  # [n, head_dim]
                rank=self.rank,
                scaling=self.scaling,
                eps=self.inverse_eps,
            )
            self.hp_prompt_cache[kv_head_idx] = {"B": B, "cache": cache}

    def normal_prefill_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float,
    ) -> torch.Tensor:
        """
        Standard causal attention for the prefill phase.

        From supervisor doc: "Prompt Tokens (Past): We don't care about
        causality within the prompt. Prompt tokens use this exact equation
        to bidirectionally understand the whole context."
        (Causal mask still applied here via attention_mask from HuggingFace.)
        """
        key_states   = repeat_kv(key_states,   self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        scores = query_states @ key_states.transpose(2, 3) * self.scaling

        if attention_mask is not None:
            scores = scores + attention_mask[:, :, :, :key_states.shape[-2]]

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

        input_shape  = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states   = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if position_embeddings is None:
            raise ValueError("position_embeddings required for RoPE.")

        # RoPE applied before caching — positional information is baked
        # into K before SVD, so B already encodes positional structure.
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        q_len   = query_states.shape[2]
        dropout = 0.0 if not self.training else self.attention_dropout

        if q_len > 1:
            # ---- PREFILL ----
            # Build compressed cache from prompt K and V — done once.
            # From supervisor doc: "we apply SVD exclusively to the prompt
            # tokens to find the core concepts of the context."
            # Prefill attention itself runs normally (exact).
            self.build_hokus_pokus_prompt_cache(key_states, value_states)
            self.key_tail_cache   = None
            self.value_tail_cache = None

            attn_output = self.normal_prefill_attention(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attention_mask=attention_mask,
                dropout=dropout,
            )

        else:
            # ---- DECODING ----
            if self.hp_prompt_cache is None:
                raise RuntimeError("Prompt cache is empty — run prefill first.")

            # Append new token to exact tail caches (local path).
            # From supervisor doc: "Local Generation Causality: standard
            # lower-triangular causal attention just for the newly generated tokens."
            if self.key_tail_cache is None:
                self.key_tail_cache   = key_states.clone()
                self.value_tail_cache = value_states.clone()
            else:
                self.key_tail_cache   = torch.cat([self.key_tail_cache,   key_states],   dim=2)
                self.value_tail_cache = torch.cat([self.value_tail_cache, value_states], dim=2)

            # Slice attention mask to tail length only — done once here,
            # not again inside hokus_pokus_decode_attention
            tail_mask = None
            if attention_mask is not None:
                tail_len  = self.key_tail_cache.shape[2]
                tail_mask = attention_mask[..., -tail_len:]

            attn_output = hokus_pokus_decode_attention(
                module=self,
                query_states=query_states,
                key_tail_cache=self.key_tail_cache,
                value_tail_cache=self.value_tail_cache,
                attention_mask=tail_mask,
                dropout=dropout,
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None