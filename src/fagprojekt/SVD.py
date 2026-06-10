from fagprojekt.model import get_kvq, get_messages, get_true_attention_values, load_model
import torch
import os
import numpy as np
import math
from pathlib import Path

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


def get_rmse(key_head, query_head, value_head, k_per_method):
    """Compute RMSE, relative Frobenius error, and cosine similarity for each method and k.

    Args:
        k_per_method: dict mapping method name to a list of k values, e.g.
            {'method_1': [80, 90], 'method_2': [95], 'method_3': [60], 'method_4': [85]}
    """
    true_attn = get_true_attention_values(query_head, key_head, value_head)

    all_results = {name: [] for name in k_per_method}

    for name, k_list in k_per_method.items():
        for k in k_list:
            if name == 'method_1':
                _, approx = method_1(key_head, query_head, value_head, k=k)
            elif name == 'method_4':
                _, approx = method_4(key_head, query_head, value_head, k=k)
            elif name == 'method_2':
                _, _, approx = method_2(key_head, query_head, value_head, k=k)
            elif name == 'method_3':
                _, _, approx = method_3(key_head, query_head, value_head, k=k)
            else:
                continue

            mse, _, cos_sim = compare_attention(true_attn, approx, name, want_print=False)
            all_results[name].append({'k': k, 'rmse': math.sqrt(mse), 'cos_sim': cos_sim})

    return all_results

def get_results_k(layer_idx, head_idx, num_tokens, thresholds_per_method, companies):
    import pandas as pd
    import matplotlib.pyplot as plt

    rows = []
    model,tokenizer = load_model()

    pages = range(1, 26)
    for company in companies:
        base_dir = Path(f"document-haystack/{company}/{company}_25Pages/Text_TextNeedles/{company}_25Pages_TextNeedles")
        for page in pages:
            page_path = f'{base_dir}_page_{page}.txt'
            messages, _, _ = get_messages(page_path, num_tokens=num_tokens)
            key_head, value_head, query_head = get_kvq(messages, layer_idx=layer_idx, head_idx=head_idx, want_print=False, model=model, tokenizer=tokenizer)
            all_results = get_rmse(key_head, query_head, value_head, thresholds_per_method)
            for method, results in all_results.items():
                for result in results:
                    rows.append({"company": company, "page": page, "method": method, "k": result["k"], "rmse": result["rmse"], "cos_sim": result["cos_sim"]})

        print(f"Done: {company}")

    df = pd.DataFrame(rows)

    tag = f"layer_{layer_idx}_head_{head_idx}_tokens_{num_tokens}"

    # Save full per-prompt results
    df.to_csv(f"reports/figures/SVD/k_tuning_all_{tag}_test.csv", index=False)
    print(f"Saved: reports/figures/SVD/k_tuning_all_{tag}_test.csv")

    # Save per-k stats across prompts
    stats_df = df.groupby(["method", "k"])["rmse"].agg(["mean", "std", "min", "max"]).reset_index()
    stats_df.to_csv(f"reports/figures/SVD/k_tuning_all_stats_{tag}_test.csv", index=False)
    print(f"Saved: reports/figures/SVD/k_tuning_all_stats_{tag}_test.csv")

    # Save best (lowest mean rmse) per method across all ks
    best_df = stats_df.loc[stats_df.groupby("method")["mean"].idxmin(), ["method", "k", "mean"]].rename(columns={"mean": "min_mean_rmse"}).reset_index(drop=True)
    best_df.to_csv(f"reports/figures/SVD/k_tuning_best_{tag}_test.csv", index=False)
    print(f"Saved: reports/figures/SVD/k_tuning_best_{tag}_test.csv")

    n_prompts = len(df) // len(df["method"].unique()) // len(df["k"].unique())

    # Plot RMSE vs k with spread across prompts
    methods = df["method"].unique()
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4), sharey=True)
    for ax, method in zip(axes, methods):
        m = stats_df[stats_df["method"] == method].sort_values("k")
        raw = df[df["method"] == method].sort_values("k")
        ax.scatter(raw["k"], raw["rmse"], s=8, alpha=0.3, color="steelblue", label="per-prompt")
        ax.plot(m["k"], m["mean"], linewidth=2, color="tomato", label="mean")
        ax.fill_between(m["k"], m["mean"] - m["std"], m["mean"] + m["std"], alpha=0.35, color="orange", label="±std")

        # Elbow: exclude near-zero plateau, normalize to [0,1], find where slope first rises above -1
        ks = m["k"].values.astype(float)
        means = m["mean"].values.astype(float)
        valid = means > means.max() * 0.01
        ks_v, means_v = ks[valid], means[valid]
        if len(ks_v) > 1:
            k_range = ks_v.max() - ks_v.min()
            r_range = means_v.max() - means_v.min()
            if k_range > 0 and r_range > 0:
                k_norm = (ks_v - ks_v.min()) / k_range
                r_norm = (means_v - means_v.min()) / r_range
                slopes_norm = np.diff(r_norm) / np.diff(k_norm)
                cross_idx = np.where(slopes_norm >= -1)[0]
                if len(cross_idx) > 0:
                    k_cross = int(ks_v[cross_idx[0]])
                    ax.axvline(k_cross, color="green", linestyle="--", linewidth=1.5, label=f"elbow k={k_cross}")

        ax.set_xlabel("k")
        ax.legend(fontsize=7)
        ax.tick_params(labelleft=True)
        ax.set_title(method, fontsize=9, fontweight="bold")
        ax.set_ylabel("RMSE")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"RMSE vs k (spread across {n_prompts} prompts) — Layer {layer_idx}, Head {head_idx}", fontsize=11)
    fig.tight_layout()
    dist_path = f"reports/figures/SVD/k_distribution_{tag}_test.pdf"
    fig.savefig(dist_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {dist_path}")

    # Plot cosine similarity vs k with spread across prompts
    cos_stats_df = df.groupby(["method", "k"])["cos_sim"].agg(["mean", "std"]).reset_index()
    fig2, axes2 = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4), sharey=True)
    for ax, method in zip(axes2, methods):
        m = cos_stats_df[cos_stats_df["method"] == method].sort_values("k")
        raw = df[df["method"] == method].sort_values("k")
        ax.scatter(raw["k"], raw["cos_sim"], s=8, alpha=0.3, color="steelblue", label="per-prompt")
        ax.plot(m["k"], m["mean"], linewidth=2, color="tomato", label="mean")
        ax.fill_between(m["k"], m["mean"] - m["std"], m["mean"] + m["std"], alpha=0.35, color="orange", label="±std")

        # Elbow: normalize to [0,1], find where slope first drops below +1
        ks = m["k"].values.astype(float)
        means = m["mean"].values.astype(float)
        k_range = ks.max() - ks.min()
        c_range = means.max() - means.min()
        if len(ks) > 1 and k_range > 0 and c_range > 0:
            # xnormalized = (x - xminimum) / range of x
            k_norm = (ks - ks.min()) / k_range
            c_norm = (means - means.min()) / c_range
            slopes_norm = np.diff(c_norm) / np.diff(k_norm)
            cross_idx = np.where(slopes_norm < 1)[0]
            if len(cross_idx) > 0:
                k_cross = int(ks[cross_idx[0]])
                ax.axvline(k_cross, color="green", linestyle="--", linewidth=1.5, label=f"elbow k={k_cross}")

        ax.set_xlabel("k")
        ax.legend(fontsize=7)
        ax.tick_params(labelleft=True)
        ax.set_title(method, fontsize=9, fontweight="bold")
        ax.set_ylabel("Cosine similarity")
        ax.grid(True, alpha=0.3)
    fig2.suptitle(f"Cosine similarity vs k (spread across {n_prompts} prompts) — Layer {layer_idx}, Head {head_idx}", fontsize=11)
    fig2.tight_layout()
    cos_path = f"reports/figures/SVD/k_cosine_{tag}_test.pdf"
    fig2.savefig(cos_path, dpi=150)
    plt.close(fig2)
    print(f"Saved: {cos_path}")


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

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    num_tokens = int(os.environ["NUM_TOKENS"])
    layer_idx  = int(os.environ["LAYER_IDX"])
    head_idx   = int(os.environ["HEAD_IDX"])

    all_methods = ['method_1', 'method_2', 'method_3', 'method_4']

    # Finding best k
    k_list = np.linspace(1, 150, 50, dtype=int).tolist()
    companies = ['Barclays','BlackRock','BNYMellon','CapitalOne','CitiGroup','Cofinimmo','CVS','DWS','Entain']
    # get_results_k(layer_idx, head_idx, num_tokens, {m: k_list for m in all_methods}, companies)

    # Testing given best k — averaged across multiple companies
    best_k = {"method_1":[15,34,63], "method_2": [15,22,75], "method_3": [15,25,69], "method_4": [15,22,63]}
    companies = ['JPMorgan','Kroger','NewRiver','PNC','Reach','Sagicor','United','UPS','Vesuvius']
    get_results_k(layer_idx, head_idx, num_tokens, best_k, companies)
