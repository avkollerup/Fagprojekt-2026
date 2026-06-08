from fagprojekt.model import get_kvq, get_messages, get_true_attention_values, load_model
import torch
import os
import numpy as np
import math

def do_SVD(matrix):
    """Compute singular value decomposition

    Args:
        matrix: Input tensor to decompose.

    Returns:
        U, singular values, and Vh from the SVD.
    """
    U, s, Vt = torch.linalg.svd(matrix, full_matrices=False)
    return U, s, Vt


def method_1(key_head, query_head, value_head, k):
    """Method 1: Decomposition of the key matrix only"""
    U_K, s_K, Vt_K = do_SVD(key_head)

    k_eff = min(k, s_K.shape[0])
    K = U_K[:, :k_eff] @ torch.diag(s_K[:k_eff]) @ Vt_K[:k_eff, :]

    # Mask: Strict upper-triangular = -inf above diagonal, 0 on/below diagonal
    M = torch.triu(torch.full((query_head.shape[0], query_head.shape[0]), float("-inf"), device=query_head.device, dtype=query_head.dtype),diagonal=1)
    d = key_head.shape[-1]
    # Compute attention values
    attn_values = torch.softmax((M + (query_head @ K.T)/ math.sqrt(d)), dim=-1) @ value_head
    return K, attn_values

def decompose_K(key_head, k):
    U_K, s_K, Vt_K = do_SVD(key_head)

    k_eff = min(k, s_K.shape[0])
    U_k = U_K[:, :k_eff]
    S_k = torch.diag(s_K[:k_eff])

    A = U_k @ S_k  
    B = Vt_K[:k_eff, :].T
    
    return A, B

def method_4(key_head, query_head, value_head, k):
    """Method 4: Decomposition of the value matrix only"""
    U_V, s_V, Vt_V = do_SVD(value_head)

    k_eff = min(k, s_V.shape[0])
    V = U_V[:, :k_eff] @ torch.diag(s_V[:k_eff]) @ Vt_V[:k_eff, :]

    # Mask: Strict upper-triangular = -inf above diagonal, 0 on/below diagonal
    M = torch.triu(torch.full((query_head.shape[0], query_head.shape[0]), float("-inf"), device=query_head.device, dtype=query_head.dtype),diagonal=1)
    d = key_head.shape[-1]
    # Compute attention values
    attn_values = torch.softmax((M + (query_head @ key_head.T)/ math.sqrt(d)), dim=-1) @ V
    return V, attn_values

def method_2(key_head, query_head, value_head, k):
    """Method 2: Decomposition of the key and value matrix seperately"""
    U_K, s_K, Vt_K = do_SVD(key_head)
    U_V, s_V, Vt_V = do_SVD(value_head)

    k_k = min(k, s_K.shape[0])
    k_v = min(k, s_V.shape[0])
    K = U_K[:, :k_k] @ torch.diag(s_K[:k_k]) @ Vt_K[:k_k, :]
    V = U_V[:, :k_v] @ torch.diag(s_V[:k_v]) @ Vt_V[:k_v, :]

    # Mask: Strict upper-triangular = -inf above diagonal, 0 on/below diagonal
    M = torch.triu(torch.full((query_head.shape[0], query_head.shape[0]), float("-inf"), device=query_head.device, dtype=query_head.dtype),diagonal=1)
    d = key_head.shape[-1]
    # Compute attention values
    attn_values = torch.softmax((M + (query_head @ K.T)/ math.sqrt(d)), dim=-1) @ V
    return K, V, attn_values

def method_3(key_head, query_head, value_head, k):
    """Method 3: Jointly decompose key and value matrix"""
    # Stack features horizontally:
    joint = torch.cat((key_head, value_head), dim=1)
    U_J, s_J, Vt_J = do_SVD(joint)

    k_eff = min(k, s_J.shape[0])
    U_k = U_J[:, :k_eff]
    S_k = torch.diag(s_J[:k_eff])
    Vt_k = Vt_J[:k_eff, :]

    # Determine A matrix
    A = U_k @ S_k  

    # Extract B, C matrix
    head_dim = key_head.shape[1]
    B = Vt_k[:, :head_dim].T
    C = Vt_k[:, head_dim:].T

    # Calculate K and V again
    K = A @ B.T
    V = A @ C.T

    # Mask: Strict upper-triangular = -inf above diagonal, 0 on/below diagonal
    M = torch.triu(torch.full((query_head.shape[0], query_head.shape[0]), float("-inf"), device=query_head.device, dtype=query_head.dtype),diagonal=1)
    d = key_head.shape[-1]
    # Compute attention values
    attn_values = torch.softmax((M + (query_head @ K.T)/ math.sqrt(d)), dim=-1) @ V
    return K, V, attn_values

def first_k_for_threshold(matrix: torch.Tensor, threshold: float = 0.9) -> int:
    """Smallest k such that the top-k singular values capture at least `threshold` of total variance."""
    singular_values = torch.linalg.svdvals(matrix.float())
    explained_variance = singular_values ** 2
    cumulative_ratio = torch.cumsum(explained_variance, dim=0) / torch.sum(explained_variance)
    indices = torch.where(cumulative_ratio >= threshold)[0]
    return int(indices[0].item()) + 1  # component needed


def tune_threshold(key_head, query_head, value_head, threshold_list):
    """Compute relative MSE (MSE / mean(true²)) at each threshold for each SVD method."""
    true_attn = get_true_attention_values(query_head, key_head, value_head)
    joint = torch.cat((key_head, value_head), dim=1)

    all_results = {'method_1': [], 'method_2': [], 'method_3': [], 'method_4': []}

    for threshold in threshold_list:
        k_k      = first_k_for_threshold(key_head,  threshold)
        k_v      = first_k_for_threshold(value_head, threshold)
        k_joint  = first_k_for_threshold(joint,      threshold)
        k_kv_sep = max(k_k, k_v)

        candidates = {
            'method_1': (k_k,      method_1(key_head, query_head, value_head, k=k_k)[1]),
            'method_4': (k_v,      method_4(key_head, query_head, value_head, k=k_v)[1]),
            'method_2': (k_kv_sep, method_2(key_head, query_head, value_head, k=k_kv_sep)[2]),
            'method_3': (k_joint,  method_3(key_head, query_head, value_head, k=k_joint)[2]),
        }

        denom = torch.mean(true_attn ** 2)
        for name, (k, approx) in candidates.items():
            mse = (torch.mean((true_attn - approx) ** 2) / denom).item()
            all_results[name].append({'threshold': threshold, 'k': k, 'rel_mse': mse})

    return all_results


def compare_attention(true_attn, approx_attn, name, want_print=True):
    """ We have used three metrics:
        - MSE for raw error (might look small because of small values)
        - Frobenius norm for scale-independent accuracy
        - Cosine similarity to capture structural (attention pattern) similarity"""

    mse = torch.mean((true_attn - approx_attn) ** 2).item()
    rel_frob = (torch.norm(true_attn - approx_attn, p="fro") / torch.norm(true_attn, p="fro")).item()
    cos = torch.nn.functional.cosine_similarity(true_attn.flatten(), approx_attn.flatten(), dim=0).item()
    if want_print:
        print(f"{name}:")
        print(f"  MSE: {mse:.6e}")
        print(f"  Relative Frobenius error: {rel_frob:.6e}")
        print(f"  Cosine similarity: {cos:.6f}\n")
    return mse, rel_frob, cos


if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    from dotenv import load_dotenv
    load_dotenv()

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    num_tokens = int(os.environ["NUM_TOKENS"])
    layer_idx  = int(os.environ["LAYER_IDX"])
    head_idx   = int(os.environ["HEAD_IDX"])

    threshold_list = np.linspace(0.5, 0.99, 50).tolist()

    rows = []

    # load model only once
    model,tokenizer = load_model()

    # iterate over pages:
    pages = range(1, 11)
    for page in pages:
        path = f'document-haystack/AIG/AIG_10Pages/Text_TextNeedles/AIG_10Pages_TextNeedles_page_{page}.txt'
        messages, _, _ = get_messages(path, num_tokens=num_tokens)
        key_head, value_head, query_head = get_kvq(messages, layer_idx=layer_idx, head_idx=head_idx, want_print=False, model=model, tokenizer=tokenizer)

        all_results = tune_threshold(key_head, query_head, value_head, threshold_list)
        for method, results in all_results.items():
            for result in results:
                rows.append({"page": page, "method": method, "threshold": result["threshold"], "k": result["k"], "rel_mse": result["rel_mse"]})

    df = pd.DataFrame(rows)

    tag = f"layer_{layer_idx}_head_{head_idx}_tokens_{num_tokens}"

    # Save full per-prompt results
    all_path = f"reports/figures/threshold_tuning_all_{tag}.csv"
    df.to_csv(all_path, index=False)
    print(f"Saved: {all_path}")

    # Save per-threshold stats across prompts
    stats_df = df.groupby(["method", "threshold"])["rel_mse"].agg(["mean", "std", "min", "max"]).reset_index()
    stats_df.to_csv(f"reports/figures/threshold_tuning_all_stats_{tag}.csv", index=False)
    print(f"Saved: reports/figures/threshold_tuning_all_stats_{tag}.csv")

    # Save best (lowest mean rel_mse) per method across all thresholds
    best_df = df.loc[df.groupby("method")["rel_mse"].idxmin(), ["method", "threshold", "rel_mse"]].rename(columns={"rel_mse": "min_rel_mse"}).reset_index(drop=True)
    best_df.to_csv(f"reports/figures/threshold_tuning_best_{tag}.csv", index=False)
    print(f"Saved: reports/figures/threshold_tuning_best_{tag}.csv")

    # Plot MSE vs threshold with spread across prompts
    methods = df["method"].unique()
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4), sharey=True)
    for ax, method in zip(axes, methods):
        m = stats_df[stats_df["method"] == method].sort_values("threshold")
        raw = df[df["method"] == method].sort_values("threshold")
        ax.scatter(raw["threshold"], raw["rel_mse"], s=8, alpha=0.3, color="steelblue", label="per-prompt")
        ax.plot(m["threshold"], m["mean"], linewidth=2, color="tomato", label="mean")
        ax.fill_between(m["threshold"], m["mean"] - m["std"], m["mean"] + m["std"], alpha=0.35, color="orange", label="±std")
        ax.tick_params(labelleft=True)
        ax.set_title(method, fontsize=9, fontweight="bold")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Relative MSE (MSE / mean(true²))")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Relative MSE vs threshold (spread across {len(pages)} prompts) — Layer {layer_idx}, Head {head_idx}", fontsize=11)
    fig.tight_layout()
    dist_path = f"reports/figures/threshold_distribution_{tag}.pdf"
    fig.savefig(dist_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {dist_path}")
