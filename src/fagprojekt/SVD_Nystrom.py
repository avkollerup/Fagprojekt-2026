import math
import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from fagprojekt.model import get_messages, get_kvq, get_true_attention_values
from fagprojekt.SVD import compare_attention


def _top_right_singular_vectors(x, k):
    """Return the top-rank right singular vectors of x [T, D] as [rank, D]."""
    _, _, vh = torch.linalg.svd(x.float(), full_matrices=False)
    r = min(k, vh.shape[0])
    return vh[:r, :]


def nystrom_attention_approx(key_head, query_head, value_head, k, eps=1e-4):
    """Nyström approximation of attention using SVD landmarks.

    Only calculates global_attention to extracted K/Q/V tensors.
    No causal mask is applied.

    Args:
        key_head:   [T, D] float32
        query_head: [T, D] float32
        value_head: [T, D] float32
        k:       number of SVD landmark directions
        eps:        regularisation strength for the landmark matrix M

    Returns:
        Approximate attention output [T, D].
    """
    d = key_head.shape[-1]

    V_K = _top_right_singular_vectors(key_head, k)    # [r, D]
    V_Q = _top_right_singular_vectors(query_head, k)  # [r, D]

    # Landmark attention matrix and landmark-to-token attention
    M = F.softmax((V_Q @ V_K.T) / math.sqrt(d), dim=-1)                  # [r, r]
    R = F.softmax((V_Q @ key_head.T) / math.sqrt(d), dim=-1)           # [r, T]

    eye = torch.eye(M.shape[-1], device=M.device, dtype=M.dtype)
    M = (1.0 - eps) * M + eps * eye

    # Precomputed value summary: M^{-1} R V
    const_cache = torch.linalg.solve(M, R @ value_head.float())  # [r, D]

    # Query-side basis weights and output
    basis_weights = F.softmax(query_head.float() @ V_K.T / math.sqrt(d), dim=-1)  # [T, r]
    return (basis_weights @ const_cache).to(key_head.dtype)


def get_rmse_companies_Nystrom(model, tokenizer, layer_idx, head_idx, num_tokens, k_list, companies, path_suffix="", want_plot=False):
    import pandas as pd
    import matplotlib.pyplot as plt

    rows = []

    pages = range(1, 26)
    for company in companies:
        base_dir = Path(f"document-haystack/{company}/{company}_25Pages/Text_TextNeedles/{company}_25Pages_TextNeedles")
        for page in pages:
            page_path = f"{base_dir}_page_{page}.txt"
            messages, _, _ = get_messages(page_path, num_tokens=num_tokens)
            key_head, value_head, query_head = get_kvq(messages, layer_idx=layer_idx, head_idx=head_idx, want_print=False, model=model, tokenizer=tokenizer)
            true_attn = get_true_attention_values(query_head, key_head, value_head)
            
            true_rms = torch.mean(true_attn ** 2).sqrt().item()
            for k in k_list:
                approx = nystrom_attention_approx(key_head, query_head, value_head, k=k)
                mse, _, _ = compare_attention(true_attn, approx, "SVD_Nystrom", want_print=False)
                rel_rmse = math.sqrt(mse) / true_rms
                rows.append({"company": company, "page": page, "k": k, "rmse": rel_rmse})

        print(f"Done: {company}")

    df = pd.DataFrame(rows)
    tag = f"layer_{layer_idx}_head_{head_idx}_tokens_{num_tokens}{path_suffix}"

    df.to_csv(f"reports/figures/SVD_Nystrom/rank_tuning_all_{tag}.csv", index=False)
    print(f"Saved: reports/figures/SVD_Nystrom/rank_tuning_all_{tag}.csv")

    stats_df = df.groupby("k")["rmse"].agg(["mean", "std", "min", "max"]).reset_index()
    stats_df.to_csv(f"reports/figures/SVD_Nystrom/rank_tuning_stats_{tag}.csv", index=False)
    print(f"Saved: reports/figures/SVD_Nystrom/rank_tuning_stats_{tag}.csv")

    best_row = stats_df.loc[stats_df["mean"].idxmin(), ["k", "mean"]].rename({"mean": "min_mean_rmse"})
    best_df = best_row.to_frame().T.reset_index(drop=True)
    best_df.to_csv(f"reports/figures/SVD_Nystrom/rank_tuning_best_{tag}.csv", index=False)
    print(f"Saved: reports/figures/SVD_Nystrom/rank_tuning_best_{tag}.csv")



