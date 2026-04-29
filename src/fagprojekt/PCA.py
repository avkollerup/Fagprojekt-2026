import torch
import numpy as np

from fagprojekt.SVD import method_1, method_2, method_3
from fagprojekt.model import load_model, get_kvq, get_messages, get_true_attention_values

from collections import defaultdict
import matplotlib.pyplot as plt

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
def pca_analysis(num_tokens):

    # load model only once
    model,tokenizer = load_model()

    # performance lists
    method_1_dict = defaultdict(lambda: defaultdict(list))
    method_2_dict = defaultdict(lambda: defaultdict(list))
    method_3_dict = defaultdict(lambda: defaultdict(list))
    explained_var_method_1_dict = defaultdict(list)
    explained_var_method_2_dict = defaultdict(list)
    explained_var_method_3_dict = defaultdict(list)
    k90_method_1_dict = defaultdict(list)
    k90_method_2_dict = defaultdict(list)
    k90_method_3_dict = defaultdict(list)
    
    # test different number of components
    components_list = list(map(int, np.linspace(10, 200, 10)))

    # iterate over pages
    pages = range(1, 5)
    for page in pages:
        path = f'document-haystack/AIG/AIG_10Pages/Text_TextNeedles/AIG_10Pages_TextNeedles_page_{page}.txt'
        messages, _, _ = get_messages(path, num_tokens=num_tokens)

        # perform for each head
        heads = range(25,30)
        for head in heads:
            # Get K, V, Q 
            key_head, value_head, query_head = get_kvq(messages, layer_idx=0, head_idx=head, want_print=True, model=model, tokenizer=tokenizer)

            # Explained variance for K, V, and KV-joint
            ev_k = cumulative_explained_variance_for_components(key_head, components_list)
            ev_v = cumulative_explained_variance_for_components(value_head, components_list)
            joint = torch.cat((key_head, value_head), dim=1)
            ev_joint = cumulative_explained_variance_for_components(joint, components_list)

            # Method 1: K decomposition only.
            explained_var_method_1_dict[head].append(ev_k)
            # Method 2: K, V decomposed separately; track shared progress as the mean of K and V.
            explained_var_method_2_dict[head].append(((np.array(ev_k) + np.array(ev_v)) / 2.0).tolist())
            # Method 3: joint decomposition.
            explained_var_method_3_dict[head].append(ev_joint)

            # How many components for 0.9 explained variance for each method
            k90_k = first_k_for_threshold(key_head, threshold=0.9)
            k90_v = first_k_for_threshold(value_head, threshold=0.9)
            k90_joint = first_k_for_threshold(joint, threshold=0.9)

            # Method 1
            k90_method_1_dict[head].append(k90_k)
            # Method 2 needs both K and V to be well represented.
            k90_method_2_dict[head].append(max(k90_k, k90_v))
            # Method 3
            k90_method_3_dict[head].append(k90_joint)

            for num_components in components_list:
                # Get attention values
                attn_values_method_1 = method_1(key_head, query_head, value_head, k=num_components)
                attn_values_method_2 = method_2(key_head, query_head, value_head, k=num_components)
                attn_values_method_3 = method_3(key_head, query_head, value_head, k=num_components)

                true_attn_values = get_true_attention_values(query_head, key_head, value_head)

                # save MSE for each method to the lists
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
        # Get average of MSE for each head across pages
        m1_avg = np.mean(np.array(list(method_1_dict[head].values())), axis=0).tolist()
        m2_avg = np.mean(np.array(list(method_2_dict[head].values())), axis=0).tolist()
        m3_avg = np.mean(np.array(list(method_3_dict[head].values())), axis=0).tolist()

        # Get average of explained variance for each head across pages
        ev_method_1_avg = np.mean(np.array(explained_var_method_1_dict[head]), axis=0).tolist()
        ev_method_2_avg = np.mean(np.array(explained_var_method_2_dict[head]), axis=0).tolist()
        ev_method_3_avg = np.mean(np.array(explained_var_method_3_dict[head]), axis=0).tolist()

        # Get average of num components for 0.9 explained variance for each head across pages
        k90_method_1_avg = int(round(float(np.mean(k90_method_1_dict[head]))))
        k90_method_2_avg = int(round(float(np.mean(k90_method_2_dict[head]))))
        k90_method_3_avg = int(round(float(np.mean(k90_method_3_dict[head]))))


        # Plotting
        # Method 1: MSE over num components
        ax = axes[row_idx, 0]
        ax.plot(components_list, m1_avg, label=f"Head {head}", marker='o', linewidth=2, markersize=6)
        ax.set_title(f'Method 1 (K) - Head {head}', fontsize=10, fontweight='bold')
        ax.set_ylabel('Averaged MSE')
        ax.set_xlabel('Num. components (k)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Method 2: MSE over num components
        ax = axes[row_idx, 1]
        ax.plot(components_list, m2_avg, label=f"Head {head}", marker='s', linewidth=2, markersize=6)
        ax.set_title(f'Method 2 (K & V sep) - Head {head}', fontsize=10, fontweight='bold')
        ax.set_ylabel('Averaged MSE')
        ax.set_xlabel('Num. components (k)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Method 3: MSE over num components
        ax = axes[row_idx, 2]
        ax.plot(components_list, m3_avg, label=f"Head {head}", marker='^', linewidth=2, markersize=6)
        ax.set_title(f'Method 3 (K & V joint) - Head {head}', fontsize=10, fontweight='bold')
        ax.set_ylabel('Averaged MSE')
        ax.set_xlabel('Num. components (k)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Method 1: Explained variance over num components
        ax = axes[row_idx, 3]
        ax.plot(components_list, ev_method_1_avg, label="Method 1", marker='o', linewidth=2, markersize=6, color="tab:orange")
        ax.axhline(0.9, linestyle=":", color="gray", linewidth=2.0, label="Threshold = 0.9")
        if components_list[0] <= k90_method_1_avg <= components_list[-1]:
            ax.axvline(k90_method_1_avg, linestyle=":", color="tab:orange", linewidth=1.5)
            ax.text(k90_method_1_avg, 0.95, f"  k={k90_method_1_avg}", fontsize=8, color="black")
        ax.set_title(f'Expl. Var Method 1 - Head {head}', fontsize=10, fontweight='bold')
        ax.set_ylabel('Avg. cumulative explained variance')
        ax.set_xlabel('Num. components (k)')
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Method 2: Explained variance over num components
        ax = axes[row_idx, 4]
        ax.plot(components_list, ev_method_2_avg, label="Method 2", marker='s', linewidth=2, markersize=6, color="tab:green")
        ax.axhline(0.9, linestyle=":", color="gray", linewidth=2.0, label="Threshold = 0.9")
        if components_list[0] <= k90_method_2_avg <= components_list[-1]:
            ax.axvline(k90_method_2_avg, linestyle=":", color="tab:green", linewidth=1.5)
            ax.text(k90_method_2_avg, 0.95, f"  k={k90_method_2_avg}", fontsize=8, color="black")
        ax.set_title(f'Expl. Var Method 2 - Head {head}', fontsize=10, fontweight='bold')
        ax.set_ylabel('Avg. cumulative explained variance')
        ax.set_xlabel('Num. components (k)')
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Method 3: Explained variance over num components
        ax = axes[row_idx, 5]
        ax.plot(components_list, ev_method_3_avg, label="Method 3", marker='^', linewidth=2, markersize=6, color="tab:blue")
        ax.axhline(0.9, linestyle=":", color="gray", linewidth=2.0, label="Threshold = 0.9")
        if components_list[0] <= k90_method_3_avg <= components_list[-1]:
            ax.axvline(k90_method_3_avg, linestyle=":", color="tab:blue", linewidth=1.5)
            ax.text(k90_method_3_avg, 0.95, f"  k={k90_method_3_avg}", fontsize=8, color="black")
        ax.set_title(f'Expl. Var Method 3 - Head {head}', fontsize=10, fontweight='bold')
        ax.set_ylabel('Avg. cumulative explained variance')
        ax.set_xlabel('Num. components (k)')
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle('PCA Analysis - Averaged Across Pages (All Heads)', fontsize=14, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig('reports/figures/pca_analysis_avg_across_pages.pdf', dpi=150)
    plt.close(fig)

            
if __name__ == "__main__":
    num_tokens = 500
    pca_analysis(num_tokens)