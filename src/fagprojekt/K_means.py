from sklearn.cluster import KMeans
import torch
from fagprojekt.model import get_messages, get_kvq, get_true_attention_values
from fagprojekt.SVD import compare_attention
from pathlib import Path
import math
import os
import numpy as np

def k_means_clustering(key_head, value_head, query_head, clusters=8):
    """ Perform K-means clustering on the key and value head representations and 
    compute attention values using the clustered representations. 
    
    Args:
        query_head: Query tensor of shape [T, C]
        value_head: Value tensor of shape [T, C]
        key_head: Key tensor of shape [T, C]
        clusters: The number of clusters to use in K-means clustering.
    Returns:
        attn_values: The attention values computed using the clustered key and value representations.
    """

    kmeans = KMeans(n_clusters=clusters)

    # concatenate key and value head for clustering
    kv_concat = torch.cat((key_head, value_head), dim=-1) # [T, 2C]

    # fit and transform to cluster space
    kmeans.fit(kv_concat.cpu().numpy())  # convert to CPU numpy array
    centroids = torch.tensor(kmeans.cluster_centers_, device=kv_concat.device, dtype=kv_concat.dtype)  # [clusters, 2C]

    # split the clustered space back into key and value components
    A,B = torch.chunk(centroids, 2, dim=-1) # [clusters, C], [clusters, C]
    
    # compute the attention values  using the clustered key and value representations
    d = query_head.shape[-1]
    attn_values = torch.softmax((query_head @ A.T) / math.sqrt(d), dim=-1) @ B

    return attn_values


def get_rmse_companies_K_means(model, tokenizer, layer_idx, head_idx, num_tokens, clusters_list, companies, path_suffix="", want_plot=False):
    import pandas as pd
    import matplotlib.pyplot as plt

    rows = []

    pages = range(1, 26)
    for company in companies:
        base_dir = Path(f"document-haystack/{company}/{company}_25Pages/Text_TextNeedles/{company}_25Pages_TextNeedles")
        for page in pages:
            page_path = f'{base_dir}_page_{page}.txt'
            messages, _, _ = get_messages(page_path, num_tokens=num_tokens)
            key_head, value_head, query_head = get_kvq(messages, layer_idx=layer_idx, head_idx=head_idx, want_print=False, model=model, tokenizer=tokenizer)
            true_attn = get_true_attention_values(query_head, key_head, value_head)

            true_rms = torch.mean(true_attn ** 2).sqrt()
            for clusters in clusters_list:
                attn_approx = k_means_clustering(key_head, value_head, query_head, clusters=clusters)
                mse, _, _ = compare_attention(true_attn, attn_approx, "K_means", want_print=False)
                rel_rmse = math.sqrt(mse) / true_rms.item()
                rows.append({"company": company, "page": page, "clusters": clusters, "rmse": rel_rmse})

        print(f"Done: {company}")

    df = pd.DataFrame(rows)

    tag = f"layer_{layer_idx}_head_{head_idx}_tokens_{num_tokens}{path_suffix}"

    df.to_csv(f"reports/figures/K_means/clusters_tuning_all_{tag}.csv", index=False)
    print(f"Saved: reports/figures/K_means/clusters_tuning_all_{tag}.csv")

    stats_df = df.groupby("clusters")["rmse"].agg(["mean", "std", "min", "max"]).reset_index()
    stats_df.to_csv(f"reports/figures/K_means/clusters_tuning_stats_{tag}.csv", index=False)
    print(f"Saved: reports/figures/K_means/clusters_tuning_stats_{tag}.csv")

    best_row = stats_df.loc[stats_df["mean"].idxmin(), ["clusters", "mean"]].rename({"mean": "min_mean_rmse"})
    best_df = best_row.to_frame().T.reset_index(drop=True)
    best_df.to_csv(f"reports/figures/K_means/clusters_tuning_best_{tag}.csv", index=False)
    print(f"Saved: reports/figures/K_means/clusters_tuning_best_{tag}.csv")

    n_prompts = len(df) // len(clusters_list)

    if want_plot:
        fig, ax = plt.subplots(figsize=(7, 4))
        m = stats_df.sort_values("clusters")
        raw = df.sort_values("clusters")
        ax.scatter(raw["clusters"], raw["rmse"], s=8, alpha=0.3, color="steelblue", label="per-prompt")
        ax.plot(m["clusters"], m["mean"], linewidth=2, color="tomato", label="mean")
        ax.fill_between(m["clusters"], m["mean"] - m["std"], m["mean"] + m["std"], alpha=0.35, color="orange", label="±std")

        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("Relative RMSE")
        ax.set_title(f"K-means Relative RMSE vs clusters (spread across {n_prompts} prompts) — Layer {layer_idx}, Head {head_idx}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        dist_path = f"reports/figures/K_means/clusters_distribution_{tag}.pdf"
        fig.savefig(dist_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {dist_path}")


if __name__ == "__main__":
    from fagprojekt.model import load_model
    from dotenv import load_dotenv
    load_dotenv()

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    num_tokens = int(os.environ["NUM_TOKENS"])
    layer_idx  = int(os.environ["LAYER_IDX"])
    head_idx   = int(os.environ["HEAD_IDX"])

    model, tokenizer = load_model()

    # Finding best cluster number
    clusters_list = np.linspace(2, 100, 30, dtype=int).tolist()
    companies = ['Barclays', 'BlackRock', 'BNYMellon', 'CapitalOne', 'CitiGroup', 'Cofinimmo', 'CVS', 'DWS', 'Entain']
    get_rmse_companies_K_means(model=model, tokenizer=tokenizer, layer_idx=layer_idx, head_idx=head_idx, num_tokens=num_tokens, clusters_list=clusters_list, companies=companies, path_suffix="", want_plot=True)


