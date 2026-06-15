import torch
import numpy as np

from fagprojekt.SVD import method_1, method_2, method_3, method_4
from fagprojekt.model import load_model, get_kvq, get_messages, get_true_attention_values

from collections import defaultdict
import matplotlib.pyplot as plt
import os



def explained_variance_ratio(matrix: torch.Tensor, components_list: list[int]) -> list[float]:
    """Explained variance ratio: σ_k² / Σσ_i² for each k in components_list."""
    singular_values = torch.linalg.svdvals(matrix.float())
    total_variance = (singular_values ** 2).sum().item()
    max_rank = singular_values.shape[0]
    return [(singular_values[k - 1].item() ** 2 / total_variance if k <= max_rank else 0.0) for k in components_list]


def cumulative_explained_variance(matrix: torch.Tensor, components_list: list[int]) -> list[float]:
    singular_values = torch.linalg.svdvals(matrix.float())
    explained_variance = singular_values**2
    total_variance = torch.sum(explained_variance)
    cumulative_ratio = torch.cumsum(explained_variance, dim=0) / total_variance
    max_rank = cumulative_ratio.shape[0]
    return [cumulative_ratio[min(k, max_rank) - 1].item() for k in components_list]


def _plot_rmse(ax, x, avg_data, title, marker, xlabel, ylim=None):
    """Plot RMSE on a log y-scale."""
    ax.plot(x, avg_data, marker=marker, linewidth=2, markersize=6)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_ylabel('RMSE')
    ax.set_xlabel(xlabel)
    ax.set_yscale('log')
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)


def _plot_explained_var_ratio(ax, components_list, avg_data, cum_data, title, marker, color, ylim=None):
    """Plot explained variance ratio σ_k² / Σσ_i² vs number of components, with true cumulative ratio on secondary y-axis."""
    ax.plot(components_list, avg_data, marker=marker, linewidth=2, markersize=6, color=color, label="Explained variance ratio")
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_ylabel('Avg. σ_k² / Σσ_i²')
    ax.set_xlabel('Num. components (k)')
    ax.set_ylim(0.0, ylim)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(components_list, cum_data, linestyle='--', linewidth=1.5, color=color, alpha=0.5, label="Cumulative")
    ax2.set_ylabel('Cumulative explained variance', fontsize=8)
    ax2.set_ylim(0.0, 1.1)
    ax2.legend(loc='upper right')

    x_arr = np.array(components_list, dtype=float)
    c_arr = np.array(cum_data)

    # 95% explained variance
    ci95 = np.where(c_arr >= 0.95)[0]
    if len(ci95) > 0:
        k_95 = int(x_arr[ci95[0]])
        ax2.axvline(k_95, color="purple", linestyle=":", linewidth=1.5, label=f"95% var k={k_95}")

    ax2.legend(loc='upper left', fontsize=7)



def svd_analysis(num_tokens, layer_idx, companies):
    """Run SVD approximation analysis across pages and heads for a given layer.
    
    Iterates over pages from different companies and 5 attention heads, computing:

    Figure 1 - fixed k sweep:
      - RMSE vs k
      - Explained variance ratio σ_k² / Σσ_i² vs k

    The results are averaged across pages"""
    print(f"Num_tokens: {num_tokens}, layer_idx: {layer_idx}")
    # load model only once
    model,tokenizer = load_model()

    # rmse_X_dict[head][(company, page)] = list of RMSE values, one per k in components_list
    rmse_k_dict = defaultdict(lambda: defaultdict(list))
    rmse_kv_sep_dict = defaultdict(lambda: defaultdict(list))
    rmse_kv_joint_dict = defaultdict(lambda: defaultdict(list))
    rmse_v_dict = defaultdict(lambda: defaultdict(list))

    # evr_X_dict[head] = list of explained variance ratios (one per page)
    evr_k_dict = defaultdict(list)
    evr_kv_sep_dict = defaultdict(list)
    evr_kv_joint_dict = defaultdict(list)
    evr_v_dict = defaultdict(list)

    # cum_ev_X_dict[head] = list of cumulative explained variance curves (one per page)
    cum_ev_k_dict = defaultdict(list)
    cum_ev_kv_sep_dict = defaultdict(list)
    cum_ev_kv_joint_dict = defaultdict(list)
    cum_ev_v_dict = defaultdict(list)

    components_list = list(map(int, np.linspace(1, 150, 25)))

    # iterate over pages
    pages = range(1, 26)
    for company in companies:
        base_dir = f"document-haystack/{company}/{company}_25Pages/Text_TextNeedles/{company}_25Pages_TextNeedles_page"
        for page in pages:
            page_path = f'{base_dir}_{page}.txt'
            messages, _, _ = get_messages(page_path, num_tokens=num_tokens)

            # perform for each head
            heads = range(0,5)
            for head in heads:
                # Get K, V, Q
                key_head, value_head, query_head = get_kvq(messages, layer_idx=layer_idx, head_idx=head, want_print=False, model=model, tokenizer=tokenizer)

                joint = torch.cat((key_head, value_head), dim=1)

                # Explained variance ratio curves (figure 1)
                evr_k = explained_variance_ratio(key_head, components_list)
                evr_v = explained_variance_ratio(value_head, components_list)
                evr_joint = explained_variance_ratio(joint, components_list)

                evr_k_dict[head].append(evr_k)
                # Method 2 decomposes K and V separately; average their curves as a shared summary
                evr_kv_sep_dict[head].append(((np.array(evr_k) + np.array(evr_v)) / 2.0).tolist())
                evr_kv_joint_dict[head].append(evr_joint)
                evr_v_dict[head].append(evr_v)

                # Cumulative explained variance curves (figure 1, secondary axis)
                cev_k = cumulative_explained_variance(key_head, components_list)
                cev_v = cumulative_explained_variance(value_head, components_list)
                cev_joint = cumulative_explained_variance(joint, components_list)

                cum_ev_k_dict[head].append(cev_k)
                cum_ev_kv_sep_dict[head].append(((np.array(cev_k) + np.array(cev_v)) / 2.0).tolist())
                cum_ev_kv_joint_dict[head].append(cev_joint)
                cum_ev_v_dict[head].append(cev_v)

                true_attn_values = get_true_attention_values(query_head, key_head, value_head)
                
                # Fixed k sweep (figure 1): same k applied to all methods
                for num_components in components_list:
                    _, attn_values_k = method_1(key_head, query_head, value_head, k=num_components)
                    _, _, attn_values_kv_sep = method_2(key_head, query_head, value_head, k=num_components)
                    _, _, attn_values_kv_joint = method_3(key_head, query_head, value_head, k=num_components)
                    _, attn_values_v = method_4(key_head, query_head, value_head, k=num_components)

                    rmse_k_dict[head][(company, page)].append(torch.mean((true_attn_values - attn_values_k) ** 2).sqrt().item())
                    rmse_kv_sep_dict[head][(company, page)].append(torch.mean((true_attn_values - attn_values_kv_sep) ** 2).sqrt().item())
                    rmse_kv_joint_dict[head][(company, page)].append(torch.mean((true_attn_values - attn_values_kv_joint) ** 2).sqrt().item())
                    rmse_v_dict[head][(company, page)].append(torch.mean((true_attn_values - attn_values_v) ** 2).sqrt().item())

        print(f"Done: {company}")

    heads = sorted(rmse_k_dict.keys())
    num_heads = len(heads)

    # ----------------------------------------------------------------
    # Figure 1: fixed k sweep - RMSE vs k and relative explained variance
    # ----------------------------------------------------------------
    fig1, axes1 = plt.subplots(num_heads, 8, figsize=(32, 4 * num_heads))
    if num_heads == 1:
        axes1 = axes1.reshape(1, -1)

    for row_idx, head in enumerate(heads):
        # Average RMSE across all (company, page) prompts for this head
        m1_avg = np.mean(np.array(list(rmse_k_dict[head].values())), axis=0).tolist()
        m4_avg = np.mean(np.array(list(rmse_v_dict[head].values())), axis=0).tolist()
        m2_avg = np.mean(np.array(list(rmse_kv_sep_dict[head].values())), axis=0).tolist()
        m3_avg = np.mean(np.array(list(rmse_kv_joint_dict[head].values())), axis=0).tolist()

        # Average explained variance ratio curves across pages for this head
        ev1_avg = np.mean(np.array(evr_k_dict[head]), axis=0).tolist()
        ev4_avg = np.mean(np.array(evr_v_dict[head]), axis=0).tolist()
        ev2_avg = np.mean(np.array(evr_kv_sep_dict[head]), axis=0).tolist()
        ev3_avg = np.mean(np.array(evr_kv_joint_dict[head]), axis=0).tolist()

        # Average cumulative explained variance curves across pages for this head
        cev1_avg = np.mean(np.array(cum_ev_k_dict[head]), axis=0).tolist()
        cev4_avg = np.mean(np.array(cum_ev_v_dict[head]), axis=0).tolist()
        cev2_avg = np.mean(np.array(cum_ev_kv_sep_dict[head]), axis=0).tolist()
        cev3_avg = np.mean(np.array(cum_ev_kv_joint_dict[head]), axis=0).tolist()

        # RMSE vs k
        all_rmse = m1_avg + m4_avg + m2_avg + m3_avg
        rmse_ylim = (min(all_rmse) * 0.5, max(all_rmse) * 2)
        _plot_rmse(axes1[row_idx, 0], components_list, m1_avg, f'RMSE - Method 1 (K) - Head {head}',           'o', 'Num. components (k)', ylim=rmse_ylim)
        _plot_rmse(axes1[row_idx, 1], components_list, m4_avg, f'RMSE - Method 4 (V) - Head {head}',           'D', 'Num. components (k)', ylim=rmse_ylim)
        _plot_rmse(axes1[row_idx, 2], components_list, m2_avg, f'RMSE - Method 2 (K & V sep) - Head {head}',   's', 'Num. components (k)', ylim=rmse_ylim)
        _plot_rmse(axes1[row_idx, 3], components_list, m3_avg, f'RMSE - Method 3 (K & V joint) - Head {head}', '^', 'Num. components (k)', ylim=rmse_ylim)

        # Explained variance ratio vs k
        ev_ylim = max(max(ev1_avg), max(ev4_avg), max(ev2_avg), max(ev3_avg)) * 1.15
        _plot_explained_var_ratio(axes1[row_idx, 4], components_list, ev1_avg, cev1_avg, f'Expl. Var Ratio - Method 1 (K) - Head {head}',           'o', 'tab:orange', ylim=ev_ylim)
        _plot_explained_var_ratio(axes1[row_idx, 5], components_list, ev4_avg, cev4_avg, f'Expl. Var Ratio - Method 4 (V) - Head {head}',           'D', 'tab:red',    ylim=ev_ylim)
        _plot_explained_var_ratio(axes1[row_idx, 6], components_list, ev2_avg, cev2_avg, f'Expl. Var Ratio - Method 2 (K & V sep) - Head {head}',   's', 'tab:green',  ylim=ev_ylim)
        _plot_explained_var_ratio(axes1[row_idx, 7], components_list, ev3_avg, cev3_avg, f'Expl. Var Ratio - Method 3 (K & V joint) - Head {head}', '^', 'tab:blue',   ylim=ev_ylim)

    n_pages = len(companies) * len(pages)
    fig1.suptitle(f'SVD Analysis (k sweep) - Layer {layer_idx} - Averaged Across {n_pages} Pages - Num_tokens {num_tokens}', fontsize=14, fontweight='bold', y=0.995)
    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    fig1.savefig(f'reports/figures/SVD/SVD_k_sweep_layer_{layer_idx}_num_tokens_{num_tokens}.pdf', dpi=150)
    plt.close(fig1)


if __name__ == "__main__":
    # --------------- PARAMETERS --------------
    from dotenv import load_dotenv
    load_dotenv()

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    num_tokens = int(os.environ["NUM_TOKENS"])
    layer_idx = int(os.environ["LAYER_IDX"])

    # --------------- SVD analysis across companies ---------------
    companies = ['Barclays','BlackRock','BNYMellon','CapitalOne','CitiGroup','Cofinimmo','CVS','DWS','Entain']
    svd_analysis(num_tokens=num_tokens, layer_idx=layer_idx, companies=companies)
