import torch
import numpy as np
from pathlib import Path

# We have used three metrics: 
# - MSE for raw error (might look small because of small values)
# - Frobenius norm for scale-independent accuracy
# - Cosine similarity to capture structural (attention pattern) similarity
def compare_attention(true_attn, approx_attn, name):
    mse = torch.mean((true_attn - approx_attn) ** 2).item()
    rel_frob = (torch.norm(true_attn - approx_attn, p="fro") / torch.norm(true_attn, p="fro")).item()
    cos = torch.nn.functional.cosine_similarity(true_attn.flatten(), approx_attn.flatten(), dim=0).item()

    print(f"{name}:")
    print(f"  MSE: {mse:.6e}")
    print(f"  Relative Frobenius error: {rel_frob:.6e}")
    print(f"  Cosine similarity: {cos:.6f}\n")


def cumulative_explained_variance_for_components(matrix: torch.Tensor, components_list: list[int]) -> list[float]:
    singular_values = torch.linalg.svdvals(matrix.float())
    explained_variance = singular_values**2
    total_variance = torch.sum(explained_variance)
    if total_variance == 0:
        return [0.0 for _ in components_list]

    cumulative_ratio = torch.cumsum(explained_variance, dim=0) / total_variance
    max_rank = cumulative_ratio.shape[0]
    return [cumulative_ratio[min(k, max_rank) - 1].item() for k in components_list]


def first_k_for_threshold(matrix: torch.Tensor, threshold: float = 0.9) -> int:
    singular_values = torch.linalg.svdvals(matrix.float())
    explained_variance = singular_values**2
    total_variance = torch.sum(explained_variance)
    if total_variance == 0:
        return 1

    cumulative_ratio = torch.cumsum(explained_variance, dim=0) / total_variance
    indices = torch.where(cumulative_ratio >= threshold)[0]
    if indices.numel() == 0:
        return cumulative_ratio.shape[0]
    return int(indices[0].item()) + 1


# do big principal component analysis
def pca_analysis():
    # imports 
    from fagprojekt.SVD import method_1,method_2,method_3
    from fagprojekt.model import (load_model, get_kvq, get_messages, get_true_attention_values)

    from collections import defaultdict
    import matplotlib.pyplot as plt

    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # load model only once
    model,tokenizer = load_model()

    # performance lists
    method_1_dict = defaultdict(lambda: defaultdict(list))
    method_2_dict = defaultdict(lambda: defaultdict(list))
    method_3_dict = defaultdict(lambda: defaultdict(list))
    explained_var_k_dict = defaultdict(list)
    explained_var_v_dict = defaultdict(list)
    explained_var_joint_dict = defaultdict(list)
    k90_k_dict = defaultdict(list)
    k90_v_dict = defaultdict(list)
    k90_joint_dict = defaultdict(list)
    
    # test different number of components
    components_list = list(map(int, np.linspace(10, 200, 10)))

    # iterate over pages
    pages = range(1, 5)
    for page in pages:
        # get respone, kv cache and attention values
        path = f'document-haystack/AIG/AIG_10Pages/Text_TextNeedles/AIG_10Pages_TextNeedles_page_{page}.txt'
        messages, prompt, needle = get_messages(path, num_tokens=100)

        # perform for each head
        heads = range (0,3)
        for head in heads:
            key_head, value_head, query_head = get_kvq(messages, layer_idx=0, head_idx=head, want_print=True, model=model, tokenizer=tokenizer)
            joint = torch.cat((key_head, value_head), dim=1)
            true_attn_values = get_true_attention_values(query_head, key_head, value_head)

            explained_var_k_dict[head].append(cumulative_explained_variance_for_components(key_head, components_list))
            explained_var_v_dict[head].append(cumulative_explained_variance_for_components(value_head, components_list))
            explained_var_joint_dict[head].append(cumulative_explained_variance_for_components(joint, components_list))

            k90_k_dict[head].append(first_k_for_threshold(key_head, threshold=0.9))
            k90_v_dict[head].append(first_k_for_threshold(value_head, threshold=0.9))
            k90_joint_dict[head].append(first_k_for_threshold(joint, threshold=0.9))

            for num_components in components_list:
                attn_values_method_1 = method_1(key_head, query_head, value_head, k=num_components)
                attn_values_method_2 = method_2(key_head, query_head, value_head, k=num_components)
                attn_values_method_3 = method_3(key_head, query_head, value_head, k=num_components)

                true_attn_values = get_true_attention_values(query_head, key_head, value_head)

                # save to the lists
                method_1_dict[head][page].append(torch.mean((true_attn_values - attn_values_method_1)**2).item())
                method_2_dict[head][page].append(torch.mean((true_attn_values - attn_values_method_2)**2).item())
                method_3_dict[head][page].append(torch.mean((true_attn_values - attn_values_method_3)**2).item())
    
    # average across pages and plot once per head
    heads = sorted(method_1_dict.keys())
    num_heads = len(heads)

    fig, axes = plt.subplots(num_heads, 6, figsize=(24, 4 * num_heads))
    if num_heads == 1:
        axes = axes.reshape(1, -1)

    for row_idx, head in enumerate(heads):
        m1_avg = np.mean(np.array(list(method_1_dict[head].values())), axis=0).tolist()
        m2_avg = np.mean(np.array(list(method_2_dict[head].values())), axis=0).tolist()
        m3_avg = np.mean(np.array(list(method_3_dict[head].values())), axis=0).tolist()
        ev_k_avg = np.mean(np.array(explained_var_k_dict[head]), axis=0).tolist()
        ev_v_avg = np.mean(np.array(explained_var_v_dict[head]), axis=0).tolist()
        ev_joint_avg = np.mean(np.array(explained_var_joint_dict[head]), axis=0).tolist()
        k90_k_avg = int(round(float(np.mean(k90_k_dict[head]))))
        k90_v_avg = int(round(float(np.mean(k90_v_dict[head]))))
        k90_joint_avg = int(round(float(np.mean(k90_joint_dict[head]))))

        ax = axes[row_idx, 0]
        ax.plot(components_list, m1_avg, label=f"Head {head}", marker='o', linewidth=2, markersize=6)
        ax.set_title(f'Method 1 (K) - Head {head}', fontsize=10, fontweight='bold')
        ax.set_ylabel('Averaged MSE')
        ax.set_xlabel('Num. components (k)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[row_idx, 1]
        ax.plot(components_list, m2_avg, label=f"Head {head}", marker='s', linewidth=2, markersize=6)
        ax.set_title(f'Method 2 (K & V sep) - Head {head}', fontsize=10, fontweight='bold')
        ax.set_ylabel('Averaged MSE')
        ax.set_xlabel('Num. components (k)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[row_idx, 2]
        ax.plot(components_list, m3_avg, label=f"Head {head}", marker='^', linewidth=2, markersize=6)
        ax.set_title(f'Method 3 (K & V joint) - Head {head}', fontsize=10, fontweight='bold')
        ax.set_ylabel('Averaged MSE')
        ax.set_xlabel('Num. components (k)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[row_idx, 3]
        ax.plot(components_list, ev_k_avg, label="K", marker='o', linewidth=2, markersize=6, color="tab:orange")
        ax.axhline(0.9, linestyle=":", color="gray", linewidth=2.0, label="Threshold = 0.9")
        if components_list[0] <= k90_k_avg <= components_list[-1]:
            ax.axvline(k90_k_avg, linestyle=":", color="tab:orange", linewidth=1.5)
            ax.text(k90_k_avg, 0.95, f"  k={k90_k_avg}", fontsize=8, color="tab:orange")
        ax.set_title(f'Expl. Var K - Head {head}', fontsize=10, fontweight='bold')
        ax.set_ylabel('Avg. cumulative explained variance')
        ax.set_xlabel('Num. components (k)')
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[row_idx, 4]
        ax.plot(components_list, ev_v_avg, label="V", marker='s', linewidth=2, markersize=6, color="tab:green")
        ax.axhline(0.9, linestyle=":", color="gray", linewidth=2.0, label="Threshold = 0.9")
        if components_list[0] <= k90_v_avg <= components_list[-1]:
            ax.axvline(k90_v_avg, linestyle=":", color="tab:green", linewidth=1.5)
            ax.text(k90_v_avg, 0.95, f"  k={k90_v_avg}", fontsize=8, color="tab:green")
        ax.set_title(f'Expl. Var V - Head {head}', fontsize=10, fontweight='bold')
        ax.set_ylabel('Avg. cumulative explained variance')
        ax.set_xlabel('Num. components (k)')
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[row_idx, 5]
        ax.plot(components_list, ev_joint_avg, label="K & V", marker='^', linewidth=2, markersize=6, color="tab:blue")
        ax.axhline(0.9, linestyle=":", color="gray", linewidth=2.0, label="Threshold = 0.9")
        if components_list[0] <= k90_joint_avg <= components_list[-1]:
            ax.axvline(k90_joint_avg, linestyle=":", color="tab:blue", linewidth=1.5)
            ax.text(k90_joint_avg, 0.95, f"  k={k90_joint_avg}", fontsize=8, color="tab:blue")
        ax.set_title(f'Expl. Var K&V - Head {head}', fontsize=10, fontweight='bold')
        ax.set_ylabel('Avg. cumulative explained variance')
        ax.set_xlabel('Num. components (k)')
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle('PCA Analysis - Averaged Across Pages (All Heads)', fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(figures_dir / 'pca_analysis_avg_across_pages.pdf', dpi=150)
    plt.close(fig)

            
if __name__ == "__main__":
    pca_analysis()