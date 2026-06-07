import torch
import numpy as np

from fagprojekt.SVD import method_1, method_2, method_3, method_4, first_k_for_threshold
from fagprojekt.model import load_model, get_kvq, get_messages, get_true_attention_values

from collections import defaultdict
import matplotlib.pyplot as plt
import os



def relative_explained_variance(matrix: torch.Tensor, components_list: list[int]) -> list[float]:
    """Relative explained variance: (σ_k / σ_1)² for each k in components_list."""
    singular_values = torch.linalg.svdvals(matrix.float())
    sigma_1_sq = singular_values[0].item() ** 2
    max_rank = singular_values.shape[0]
    # k may exceed the matrix rank (e.g. components_list reaches 200 but key_head has 128 rows)
    return [(singular_values[k - 1].item() ** 2 / sigma_1_sq if k <= max_rank else 0.0) for k in components_list]


def _plot_relative_mse(ax, x, avg_data, title, marker, xlabel, ylim=None):
    """Plot relative MSE on a log y-scale."""
    ax.plot(x, avg_data, marker=marker, linewidth=2, markersize=6)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_ylabel('Relative MSE (MSE / mean(true²))')
    ax.set_xlabel(xlabel)
    ax.set_yscale('log')
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)


def _plot_relative_explained_var(ax, components_list, avg_data, title, marker, color):
    """Plot relative explained variance (σ_k / σ_1)² vs number of components."""
    ax.plot(components_list, avg_data, marker=marker, linewidth=2, markersize=6, color=color)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_ylabel('Avg. (σ_k / σ_1)²')
    ax.set_xlabel('Num. components (k)')
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)


def _plot_k_needed(ax, threshold_list, avg_data, title, marker, color, ylim=None):
    """Plot components needed to meet each threshold."""
    ax.plot(threshold_list, avg_data, marker=marker, linewidth=2, markersize=6, color=color)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_ylabel('Avg. components needed (k)')
    ax.set_xlabel('Threshold')
    ax.grid(True, alpha=0.3)
    if ylim is not None:
        ax.set_ylim(ylim)


def pca_analysis(num_tokens, layer_idx):
    """Run SVD approximation analysis across pages and heads for a given layer.

    Iterates over 10 pages and 5 attention heads, computing two parallel analyses:

    Figure 1 - fixed k sweep:
      - Relative MSE vs k
      - Relative explained variance (σ_k / σ_1)² vs k

    Figure 2 - threshold sweep:
      - Components needed vs threshold

    Both figures average results across pages and save one row per head to PDF.
    """
    print(f"Num_tokens: {num_tokens}, layer_idx: {layer_idx}")

    # load model only once
    model,tokenizer = load_model()

    # mse_X_dict[head][page] = list of relative MSE values, one per k in components_list
    mse_k_dict = defaultdict(lambda: defaultdict(list))
    mse_kv_sep_dict = defaultdict(lambda: defaultdict(list))
    mse_kv_joint_dict = defaultdict(lambda: defaultdict(list))
    mse_v_dict = defaultdict(lambda: defaultdict(list))

    # rel_ev_X_dict[head] = list of relative explained variance curves (one per page)
    rel_ev_k_dict = defaultdict(list)
    rel_ev_kv_sep_dict = defaultdict(list)
    rel_ev_kv_joint_dict = defaultdict(list)
    rel_ev_v_dict = defaultdict(list)

    # k_needed_X_dict[head][page] = components needed to meet each threshold, one per threshold
    k_needed_k_dict = defaultdict(lambda: defaultdict(list))
    k_needed_kv_sep_dict = defaultdict(lambda: defaultdict(list))
    k_needed_kv_joint_dict = defaultdict(lambda: defaultdict(list))
    k_needed_v_dict = defaultdict(lambda: defaultdict(list))

    # test different number of components and different thresholds
    components_list = list(map(int, np.linspace(10, 200, 10)))
    threshold_list = np.linspace(0.5, 0.99, 15).tolist()

    # iterate over pages
    pages = range(1, 11)
    for page in pages:
        path = f'document-haystack/AIG/AIG_10Pages/Text_TextNeedles/AIG_10Pages_TextNeedles_page_{page}.txt'
        messages, _, _ = get_messages(path, num_tokens=num_tokens)

        # perform for each head
        heads = range(0,5)
        for head in heads:
            # Get K, V, Q
            key_head, value_head, query_head = get_kvq(messages, layer_idx=layer_idx, head_idx=head, want_print=False, model=model, tokenizer=tokenizer)

            joint = torch.cat((key_head, value_head), dim=1)

            # Relative explained variance curves (figure 1)
            rev_k = relative_explained_variance(key_head, components_list)
            rev_v = relative_explained_variance(value_head, components_list)
            rev_joint = relative_explained_variance(joint, components_list)

            rel_ev_k_dict[head].append(rev_k)
            # Method 2 decomposes K and V separately; average their curves as a shared summary
            rel_ev_kv_sep_dict[head].append(((np.array(rev_k) + np.array(rev_v)) / 2.0).tolist())
            rel_ev_kv_joint_dict[head].append(rev_joint)
            rel_ev_v_dict[head].append(rev_v)

            true_attn_values = get_true_attention_values(query_head, key_head, value_head)
            # Normalize by mean squared true attention values so errors are comparable across heads with different scales
            denom = torch.mean(true_attn_values ** 2)

            # Fixed k sweep (figure 1): same k applied to all methods
            for num_components in components_list:
                _, attn_values_k = method_1(key_head, query_head, value_head, k=num_components)
                _, _, attn_values_kv_sep = method_2(key_head, query_head, value_head, k=num_components)
                _, _, attn_values_kv_joint = method_3(key_head, query_head, value_head, k=num_components)
                _, attn_values_v = method_4(key_head, query_head, value_head, k=num_components)

                mse_k_dict[head][page].append((torch.mean((true_attn_values - attn_values_k) ** 2) / denom).item())
                mse_kv_sep_dict[head][page].append((torch.mean((true_attn_values - attn_values_kv_sep) ** 2) / denom).item())
                mse_kv_joint_dict[head][page].append((torch.mean((true_attn_values - attn_values_kv_joint) ** 2) / denom).item())
                mse_v_dict[head][page].append((torch.mean((true_attn_values - attn_values_v) ** 2) / denom).item())

            # Threshold sweep (figure 2): k is derived per method based on the threshold
            for threshold in threshold_list:
                k_k = first_k_for_threshold(key_head, threshold)
                k_v = first_k_for_threshold(value_head, threshold)
                k_joint = first_k_for_threshold(joint, threshold)
                # Method 2 decomposes K and V separately, so both must independently meet the threshold
                k_kv_sep = max(k_k, k_v)

                k_needed_k_dict[head][page].append(k_k)
                k_needed_kv_sep_dict[head][page].append(k_kv_sep)
                k_needed_kv_joint_dict[head][page].append(k_joint)
                k_needed_v_dict[head][page].append(k_v)

    heads = sorted(mse_k_dict.keys())
    num_heads = len(heads)

    # ----------------------------------------------------------------
    # Figure 1: fixed k sweep - MSE vs k and relative explained variance
    # ----------------------------------------------------------------
    fig1, axes1 = plt.subplots(num_heads, 8, figsize=(32, 4 * num_heads))
    if num_heads == 1:
        axes1 = axes1.reshape(1, -1)

    for row_idx, head in enumerate(heads):
        # Average relative MSE across pages for this head
        m1_avg = np.mean(np.array(list(mse_k_dict[head].values())), axis=0).tolist()
        m4_avg = np.mean(np.array(list(mse_v_dict[head].values())), axis=0).tolist()
        m2_avg = np.mean(np.array(list(mse_kv_sep_dict[head].values())), axis=0).tolist()
        m3_avg = np.mean(np.array(list(mse_kv_joint_dict[head].values())), axis=0).tolist()

        # Average relative explained variance curves across pages for this head
        ev1_avg = np.mean(np.array(rel_ev_k_dict[head]), axis=0).tolist()
        ev4_avg = np.mean(np.array(rel_ev_v_dict[head]), axis=0).tolist()
        ev2_avg = np.mean(np.array(rel_ev_kv_sep_dict[head]), axis=0).tolist()
        ev3_avg = np.mean(np.array(rel_ev_kv_joint_dict[head]), axis=0).tolist()

        all_mse = m1_avg + m4_avg + m2_avg + m3_avg
        mse_ylim = (min(all_mse) * 0.5, max(all_mse) * 2)

        # Relative MSE vs k
        _plot_relative_mse(axes1[row_idx, 0], components_list, m1_avg, f'Rel. MSE - Method 1 (K) - Head {head}',           'o', 'Num. components (k)', ylim=mse_ylim)
        _plot_relative_mse(axes1[row_idx, 1], components_list, m4_avg, f'Rel. MSE - Method 4 (V) - Head {head}',           'D', 'Num. components (k)', ylim=mse_ylim)
        _plot_relative_mse(axes1[row_idx, 2], components_list, m2_avg, f'Rel. MSE - Method 2 (K & V sep) - Head {head}',   's', 'Num. components (k)', ylim=mse_ylim)
        _plot_relative_mse(axes1[row_idx, 3], components_list, m3_avg, f'Rel. MSE - Method 3 (K & V joint) - Head {head}', '^', 'Num. components (k)', ylim=mse_ylim)

        # Relative explained variance vs k
        _plot_relative_explained_var(axes1[row_idx, 4], components_list, ev1_avg, f'Rel. Explained Var - Method 1 (K) - Head {head}',           'o', 'tab:orange')
        _plot_relative_explained_var(axes1[row_idx, 5], components_list, ev4_avg, f'Rel. Explained Var - Method 4 (V) - Head {head}',           'D', 'tab:red')
        _plot_relative_explained_var(axes1[row_idx, 6], components_list, ev2_avg, f'Rel. Explained Var - Method 2 (K & V sep) - Head {head}',   's', 'tab:green')
        _plot_relative_explained_var(axes1[row_idx, 7], components_list, ev3_avg, f'Rel. Explained Var - Method 3 (K & V joint) - Head {head}', '^', 'tab:blue')

    fig1.suptitle(f'PCA Analysis (k sweep) - Layer {layer_idx} - Averaged Across Pages - Num_tokens {num_tokens}', fontsize=14, fontweight='bold', y=0.995)
    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    fig1.savefig(f'reports/figures/pca_k_sweep_layer_{layer_idx}_num_tokens_{num_tokens}.pdf', dpi=150)
    plt.close(fig1)

    # -----------------------------------------------------------------------
    # Figure 2: threshold sweep - Components needed vs threshold
    # -----------------------------------------------------------------------
    fig2, axes2 = plt.subplots(num_heads, 4, figsize=(16, 4 * num_heads))
    if num_heads == 1:
        axes2 = axes2.reshape(1, -1)

    for row_idx, head in enumerate(heads):
        # Average k needed across pages for this head
        k1_avg = np.mean(np.array(list(k_needed_k_dict[head].values())), axis=0).tolist()
        k4_avg = np.mean(np.array(list(k_needed_v_dict[head].values())), axis=0).tolist()
        k2_avg = np.mean(np.array(list(k_needed_kv_sep_dict[head].values())), axis=0).tolist()
        k3_avg = np.mean(np.array(list(k_needed_kv_joint_dict[head].values())), axis=0).tolist()

        all_k_vals = k1_avg + k4_avg + k2_avg + k3_avg
        k_needed_ylim = (max(0, min(all_k_vals) - 2), max(all_k_vals) + 2)

        # Components needed vs threshold
        _plot_k_needed(axes2[row_idx, 0], threshold_list, k1_avg, f'Components needed - Method 1 (K) - Head {head}',           'o', 'tab:orange', ylim=k_needed_ylim)
        _plot_k_needed(axes2[row_idx, 1], threshold_list, k4_avg, f'Components needed - Method 4 (V) - Head {head}',           'D', 'tab:red',    ylim=k_needed_ylim)
        _plot_k_needed(axes2[row_idx, 2], threshold_list, k2_avg, f'Components needed - Method 2 (K & V sep) - Head {head}',   's', 'tab:green',  ylim=k_needed_ylim)
        _plot_k_needed(axes2[row_idx, 3], threshold_list, k3_avg, f'Components needed - Method 3 (K & V joint) - Head {head}', '^', 'tab:blue',   ylim=k_needed_ylim)

    fig2.suptitle(f'PCA Analysis (threshold sweep) - Layer {layer_idx} - Averaged Across Pages - Num_tokens {num_tokens}', fontsize=14, fontweight='bold', y=0.995)
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    fig2.savefig(f'reports/figures/pca_threshold_sweep_layer_{layer_idx}_num_tokens_{num_tokens}.pdf', dpi=150)
    plt.close(fig2)


if __name__ == "__main__":
    # --------------- PARAMETERS --------------
    from dotenv import load_dotenv
    load_dotenv()

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    num_tokens = int(os.environ["NUM_TOKENS"])
    layer_idx = int(os.environ["LAYER_IDX"])

    # --------------- PCA analysis ---------------
    pca_analysis(num_tokens=num_tokens, layer_idx=layer_idx)
