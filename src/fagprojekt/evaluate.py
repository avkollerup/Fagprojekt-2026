import pandas as pd
from fagprojekt.model import load_model
from fagprojekt.SVD import get_rmse_companies_SVD
from fagprojekt.Hokus_pokus import get_rmse_companies_Hokus_Pokus
from fagprojekt.K_means import get_rmse_companies_K_means
import os
import itertools
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from torch.profiler import profile, ProfilerActivity, record_function, schedule
import torch
prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],schedule = schedule(wait=0,warmup=0,active=1),profile_memory=True, record_shapes=True, acc_events=True) 


def run_evaluation_rmse(model, tokenizer, layer_idx, head_idx, num_tokens, train_companies, test_companies, num_epochs):
    # ------ SVD ------
    best_k_sweep = {"method_1": [33, 45, 62], "method_2": [21, 45, 68], "method_3": [37, 45, 68], "method_4": [21, 45, 74]}
    get_rmse_companies_SVD(model=model, tokenizer=tokenizer, layer_idx=layer_idx, head_idx=head_idx, num_tokens=num_tokens, thresholds_per_method=best_k_sweep, companies=test_companies, path_suffix="_test", want_plot=False)
    df = pd.read_csv(f"reports/figures/SVD/k_tuning_all_layer_{layer_idx}_head_{head_idx}_tokens_{num_tokens}_test.csv")

    best_k_svd = {"method_1": 62, "method_2": 68, "method_3": 68, "method_4": 74}
    rmse_per_SVD_method = {
        method: df[(df["method"] == method) & (df["k"] == k)]
                .sort_values(["company", "page"])["rmse"].values
        for method, k in best_k_svd.items()
    }

    # ------ Hokus Pokus ------
    k_hp = 45
    rmse_Hokus_pokus = get_rmse_companies_Hokus_Pokus(model=model, tokenizer=tokenizer, num_tokens=num_tokens, layer_idx=layer_idx, head_idx=head_idx, k=k_hp, train_companies=train_companies, test_companies=test_companies, num_epochs=num_epochs)

    # ------ K-means ------
    clusters = [8]
    get_rmse_companies_K_means(model=model, tokenizer=tokenizer, layer_idx=layer_idx, head_idx=head_idx, num_tokens=num_tokens, clusters_list=clusters, companies=test_companies, path_suffix="_test")
    df_km = pd.read_csv(f"reports/figures/K_means/clusters_tuning_all_layer_{layer_idx}_head_{head_idx}_tokens_{num_tokens}_test.csv")
    rmse_K_means = df_km[df_km["clusters"] == 8].sort_values(["company", "page"])["rmse"].values

    # ------ Statistical tests ------
    all_rmse = {
        "method_1 (K)": rmse_per_SVD_method["method_1"],
        "method_2 (V)": rmse_per_SVD_method["method_4"],
        "method_3 (K & V sep)" : rmse_per_SVD_method["method_2"],
        "method_4 (K & V joint)": rmse_per_SVD_method["method_3"],
        "Hokus Pokus": rmse_Hokus_pokus,
        "K-means": rmse_K_means,
    }

    for name, rmse in all_rmse.items():
        print(f"{name}: mean RMSE={np.mean(rmse):.6e}, std={np.std(rmse):.6e}")
        print()

    pairs = list(itertools.combinations(all_rmse.keys(), 2))
    stats = [wilcoxon(all_rmse[a], all_rmse[b]) for a, b in pairs]
    raw_p = [s.pvalue for s in stats]
    reject, p_corrected, _, _ = multipletests(raw_p, method="bonferroni")

    for (a, b), p_raw, p_adj, rej in zip(pairs, raw_p, p_corrected, reject):
        print(f"{a} vs {b}: p_raw={p_raw:.4e}, p_adj={p_adj:.4e}, reject H0={rej}")

    return all_rmse


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    prof.start()

    num_tokens = int(os.environ["NUM_TOKENS"])
    layer_idx = int(os.environ["LAYER_IDX"])
    head_idx = int(os.environ["HEAD_IDX"])
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    num_epochs = 90
    
    model, tokenizer = load_model(want_print=False)

    train_companies = sorted(['GoldmanSachs', 'HSBC', 'JPMorgan', 'Kroger', 'NewRiver', 'PNC', 'Reach', 'Sagicor', 'United', 'UPS', 'Vesuvius', 'WoltersKluwer'])
    test_companies  = sorted(['AIG', 'AmericanAirlines', 'APA'])

    run_evaluation_rmse(model, tokenizer, layer_idx, head_idx, num_tokens, train_companies, test_companies, num_epochs)

    torch.cuda.syncronize()
    prof.step()
    prof.stop()

    with open("eval_metrics.txt", "a") as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


# 90 epochs, {"method_1": 33, "method_2": 21, "method_3": 37, "method_4": 21}
"""
method_1 (K): mean RMSE=2.784065e-01, std=3.332339e-02

method_2 (V): mean RMSE=2.976462e-01, std=2.407543e-02

method_3 (K & V sep): mean RMSE=4.009061e-01, std=3.512433e-02

method_4 (K & V joint): mean RMSE=3.294086e-01, std=3.304040e-02

Hokus Pokus: mean RMSE=1.555040e-01, std=1.884069e-02

K-means: mean RMSE=7.932307e-01, std=7.068266e-02

method_1 (K) vs method_2 (V): p_raw=1.2917e-08, p_adj=1.9375e-07, reject H0=True
method_1 (K) vs method_3 (K & V sep): p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
method_1 (K) vs method_4 (K & V joint): p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
method_1 (K) vs Hokus Pokus: p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
method_1 (K) vs K-means: p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
method_2 (V) vs method_3 (K & V sep): p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
method_2 (V) vs method_4 (K & V joint): p_raw=4.3268e-13, p_adj=6.4902e-12, reject H0=True
method_2 (V) vs Hokus Pokus: p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
method_2 (V) vs K-means: p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
method_3 (K & V sep) vs method_4 (K & V joint): p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
method_3 (K & V sep) vs Hokus Pokus: p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
method_3 (K & V sep) vs K-means: p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
method_4 (K & V joint) vs Hokus Pokus: p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
method_4 (K & V joint) vs K-means: p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
Hokus Pokus vs K-means: p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True

"""

# 90 epochs, {"method_1": 45, "method_2": 45, "method_3": 45, "method_4": 45}
"""
method_1 (K): mean RMSE=2.388932e-01, std=3.061739e-02

method_2 (V): mean RMSE=1.808087e-01, std=1.736752e-02

method_3 (K & V sep): mean RMSE=2.812025e-01, std=2.937719e-02

method_4 (K & V joint): mean RMSE=3.077195e-01, std=3.257263e-02

Hokus Pokus: mean RMSE=1.552060e-01, std=1.771987e-02

K-means: mean RMSE=7.893306e-01, std=6.517202e-02

method_1 (K) vs method_2 (V): p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
method_1 (K) vs method_3 (K & V sep): p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
method_1 (K) vs method_4 (K & V joint): p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
method_1 (K) vs Hokus Pokus: p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
method_1 (K) vs K-means: p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
method_2 (V) vs method_3 (K & V sep): p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
method_2 (V) vs method_4 (K & V joint): p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
method_2 (V) vs Hokus Pokus: p_raw=2.6199e-12, p_adj=3.9299e-11, reject H0=True
method_2 (V) vs K-means: p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
method_3 (K & V sep) vs method_4 (K & V joint): p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
method_3 (K & V sep) vs Hokus Pokus: p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
method_3 (K & V sep) vs K-means: p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
method_4 (K & V joint) vs Hokus Pokus: p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
method_4 (K & V joint) vs K-means: p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
Hokus Pokus vs K-means: p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
"""


# 90 epochs, {"method_1": 62, "method_2": 68, "method_3": 68, "method_4": 74}
"""
method_1 (K): mean RMSE=1.822175e-01, std=2.861437e-02

method_2 (V): mean RMSE=1.011422e-01, std=1.340265e-02

method_3 (K & V sep): mean RMSE=1.926945e-01, std=2.587086e-02

method_4 (K & V joint): mean RMSE=2.477381e-01, std=3.014462e-02

Hokus Pokus: mean RMSE=1.525799e-01, std=1.678583e-02

K-means: mean RMSE=7.927416e-01, std=6.988793e-02

method_1 (K) vs method_2 (V): p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
method_1 (K) vs method_3 (K & V sep): p_raw=2.4161e-09, p_adj=3.6242e-08, reject H0=True
method_1 (K) vs method_4 (K & V joint): p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
method_1 (K) vs Hokus Pokus: p_raw=8.0127e-12, p_adj=1.2019e-10, reject H0=True
method_1 (K) vs K-means: p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
method_2 (V) vs method_3 (K & V sep): p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
method_2 (V) vs method_4 (K & V joint): p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
method_2 (V) vs Hokus Pokus: p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
method_2 (V) vs K-means: p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
method_3 (K & V sep) vs method_4 (K & V joint): p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
method_3 (K & V sep) vs Hokus Pokus: p_raw=1.0881e-13, p_adj=1.6321e-12, reject H0=True
method_3 (K & V sep) vs K-means: p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
method_4 (K & V joint) vs Hokus Pokus: p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
method_4 (K & V joint) vs K-means: p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
Hokus Pokus vs K-means: p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
"""