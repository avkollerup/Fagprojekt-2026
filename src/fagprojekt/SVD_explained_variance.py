import torch
import numpy as np

from fagprojekt.model import load_model, get_kvq, get_messages

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

    ax2.legend(loc='upper right', fontsize=7)



def svd_analysis(num_tokens, layer_idx, companies, head_idx=None):
    """Run SVD approximation analysis across pages for a given layer and head.

    Figure layout (1 row × 4 columns):
      Row 0 - explained variance ratio σ_k² / Σσ_i² vs k for each method

    The results are averaged across pages"""
    print(f"Num_tokens: {num_tokens}, layer_idx: {layer_idx}, head_idx: {head_idx}")
    # load model only once
    model,tokenizer = load_model()

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

    components_list = list(map(int, np.linspace(1, 300, 50)))


    # iterate over pages
    pages = range(1, 26)
    for company in companies:
        base_dir = f"document-haystack/{company}/{company}_25Pages/Text_TextNeedles/{company}_25Pages_TextNeedles_page"
        for page in pages:
            page_path = f'{base_dir}_{page}.txt'
            messages, _, _ = get_messages(page_path, num_tokens=num_tokens)
            # Get K, V, Q
            key_head, value_head, _ = get_kvq(messages, layer_idx=layer_idx, head_idx=head_idx, want_print=False, model=model, tokenizer=tokenizer)

            joint = torch.cat((key_head, value_head), dim=1)

            # Explained variance ratio curves (figure 1)
            evr_k = explained_variance_ratio(key_head, components_list)
            evr_v = explained_variance_ratio(value_head, components_list)
            evr_joint = explained_variance_ratio(joint, components_list)

            evr_k_dict[head_idx].append(evr_k)
            # Method 2 decomposes K and V separately; average their curves as a shared summary
            evr_kv_sep_dict[head_idx].append(((np.array(evr_k) + np.array(evr_v)) / 2.0).tolist())
            evr_kv_joint_dict[head_idx].append(evr_joint)
            evr_v_dict[head_idx].append(evr_v)

            # Cumulative explained variance curves (figure 1, secondary axis)
            cev_k = cumulative_explained_variance(key_head, components_list)
            cev_v = cumulative_explained_variance(value_head, components_list)
            cev_joint = cumulative_explained_variance(joint, components_list)

            cum_ev_k_dict[head_idx].append(cev_k)
            cum_ev_kv_sep_dict[head_idx].append(((np.array(cev_k) + np.array(cev_v)) / 2.0).tolist())
            cum_ev_kv_joint_dict[head_idx].append(cev_joint)
            cum_ev_v_dict[head_idx].append(cev_v)

        print(f"Done: {company}")

    n_pages = len(companies) * len(pages)

    ev1_avg = np.mean(np.array(evr_k_dict[head_idx]), axis=0).tolist()
    ev4_avg = np.mean(np.array(evr_v_dict[head_idx]), axis=0).tolist()
    ev2_avg = np.mean(np.array(evr_kv_sep_dict[head_idx]), axis=0).tolist()
    ev3_avg = np.mean(np.array(evr_kv_joint_dict[head_idx]), axis=0).tolist()

    cev1_avg = np.mean(np.array(cum_ev_k_dict[head_idx]), axis=0).tolist()
    cev4_avg = np.mean(np.array(cum_ev_v_dict[head_idx]), axis=0).tolist()
    cev2_avg = np.mean(np.array(cum_ev_kv_sep_dict[head_idx]), axis=0).tolist()
    cev3_avg = np.mean(np.array(cum_ev_kv_joint_dict[head_idx]), axis=0).tolist()

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    ev_ylim = max(max(ev1_avg), max(ev4_avg), max(ev2_avg), max(ev3_avg)) * 1.15
    _plot_explained_var_ratio(axes[0], components_list, ev1_avg, cev1_avg, f'Expl. Var Ratio - Method 1 (K)',           'o', 'tab:orange', ylim=ev_ylim)
    _plot_explained_var_ratio(axes[1], components_list, ev4_avg, cev4_avg, f'Expl. Var Ratio - Method 2 (V)',           'D', 'tab:red',    ylim=ev_ylim)
    _plot_explained_var_ratio(axes[2], components_list, ev2_avg, cev2_avg, f'Expl. Var Ratio - Method 3 (K & V sep)',   's', 'tab:green',  ylim=ev_ylim)
    _plot_explained_var_ratio(axes[3], components_list, ev3_avg, cev3_avg, f'Expl. Var Ratio - Method 4 (K & V joint)', '^', 'tab:blue',   ylim=ev_ylim)

    fig.suptitle(f'SVD Analysis (k sweep) — Layer {layer_idx}, Head {head_idx} — Averaged Across {n_pages} Pages — Num_tokens {num_tokens}', fontsize=13, fontweight='bold', y=0.995,)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f'reports/figures/SVD/SVD_k_sweep_layer_{layer_idx}_head_{head_idx}_num_tokens_{num_tokens}.pdf', dpi=150)
    plt.close(fig)
    print(f"Saved: reports/figures/SVD/SVD_k_sweep_layer_{layer_idx}_head_{head_idx}_num_tokens_{num_tokens}.pdf")


if __name__ == "__main__":
    # --------------- PARAMETERS --------------
    from dotenv import load_dotenv
    load_dotenv()

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    num_tokens = int(os.environ["NUM_TOKENS"])
    layer_idx = int(os.environ["LAYER_IDX"])
    head_idx = int(os.environ["HEAD_IDX"])

    # --------------- SVD analysis across companies ---------------
    companies = ['Barclays','BlackRock','BNYMellon','CapitalOne','CitiGroup','Cofinimmo','CVS','DWS','Entain']
    svd_analysis(num_tokens=num_tokens, layer_idx=layer_idx, companies=companies, head_idx=head_idx)
