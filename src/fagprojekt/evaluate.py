import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pandas as pd
from fagprojekt.model import load_model
from fagprojekt.SVD import get_rmse_companies_SVD
from fagprojekt.Hokus_pokus import get_rmse_companies_Hokus_Pokus
from fagprojekt.K_means import get_rmse_companies_K_means
from fagprojekt.SVD_Nystrom import get_rmse_companies_Nystrom
import itertools
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


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
    clusters = 45
    get_rmse_companies_K_means(model=model, tokenizer=tokenizer, layer_idx=layer_idx, head_idx=head_idx, num_tokens=num_tokens, clusters_list=[clusters], companies=test_companies, path_suffix="_test")
    df_km = pd.read_csv(f"reports/figures/K_means/clusters_tuning_all_layer_{layer_idx}_head_{head_idx}_tokens_{num_tokens}_test.csv")
    rmse_K_means = df_km[df_km["clusters"] == clusters].sort_values(["company", "page"])["rmse"].values

    # ------ SVD Nyström ------
    k_Nystrom = 1
    get_rmse_companies_Nystrom(model=model, tokenizer=tokenizer, layer_idx=layer_idx, head_idx=head_idx, num_tokens=num_tokens, k_list=[k_Nystrom], companies=test_companies, path_suffix="_test")
    df_nystrom = pd.read_csv(f"reports/figures/SVD_Nystrom/rank_tuning_all_layer_{layer_idx}_head_{head_idx}_tokens_{num_tokens}_test.csv")
    rmse_Nystrom = df_nystrom[df_nystrom["k"] == k_Nystrom].sort_values(["company", "page"])["rmse"].values

    # ------ Statistical tests ------
    print(f"SVD k: {best_k_svd}")
    print(f"Hokus Pokus k: {k_hp}")
    print(f"K-means c: {clusters}")
    print(f"SVD Nyström k: {k_Nystrom}")

    all_rmse = {
        "method_1 (K)": rmse_per_SVD_method["method_1"],
        "method_2 (V)": rmse_per_SVD_method["method_4"],
        "method_3 (K & V sep)" : rmse_per_SVD_method["method_2"],
        "method_4 (K & V joint)": rmse_per_SVD_method["method_3"],
        "Hokus Pokus": rmse_Hokus_pokus,
        "K-means": rmse_K_means,
        "SVD Nyström": rmse_Nystrom,
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

    num_tokens = int(os.environ["NUM_TOKENS"])
    layer_idx = int(os.environ["LAYER_IDX"])
    head_idx = int(os.environ["HEAD_IDX"])
    num_epochs = 90
    
    model, tokenizer = load_model(want_print=False)

    train_companies = sorted(['GoldmanSachs', 'HSBC', 'JPMorgan', 'Kroger', 'NewRiver', 'PNC', 'Reach', 'Sagicor', 'United', 'UPS', 'Vesuvius', 'WoltersKluwer'])
    test_companies  = sorted(['AIG', 'AmericanAirlines', 'APA'])

    run_evaluation_rmse(model, tokenizer, layer_idx, head_idx, num_tokens, train_companies, test_companies, num_epochs)



# 90 epochs - Elbow k
"""
SVD k: {'method_1': 33, 'method_2': 21, 'method_3': 37, 'method_4': 21}
Hokus Pokus k: 45
K-means c: 8
SVD Nyström k: 1
method_1 (K): mean RMSE=2.794447e-01, std=3.383524e-02

method_2 (V): mean RMSE=2.983330e-01, std=2.520924e-02

method_3 (K & V sep): mean RMSE=4.015721e-01, std=3.614047e-02

method_4 (K & V joint): mean RMSE=3.300683e-01, std=3.388908e-02

Hokus Pokus: mean RMSE=1.539827e-01, std=1.718836e-02

K-means: mean RMSE=7.957047e-01, std=7.707829e-02

SVD Nyström: mean RMSE=7.589384e-01, std=2.631233e-02

method_1 (K) vs method_2 (V): p_raw=2.6912e-08, p_adj=5.6514e-07, reject H0=True
method_1 (K) vs method_3 (K & V sep): p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_1 (K) vs method_4 (K & V joint): p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_1 (K) vs Hokus Pokus: p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_1 (K) vs K-means: p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_1 (K) vs SVD Nyström: p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_2 (V) vs method_3 (K & V sep): p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_2 (V) vs method_4 (K & V joint): p_raw=4.6772e-13, p_adj=9.8221e-12, reject H0=True
method_2 (V) vs Hokus Pokus: p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_2 (V) vs K-means: p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_2 (V) vs SVD Nyström: p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_3 (K & V sep) vs method_4 (K & V joint): p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_3 (K & V sep) vs Hokus Pokus: p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_3 (K & V sep) vs K-means: p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_3 (K & V sep) vs SVD Nyström: p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_4 (K & V joint) vs Hokus Pokus: p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_4 (K & V joint) vs K-means: p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_4 (K & V joint) vs SVD Nyström: p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
Hokus Pokus vs K-means: p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
Hokus Pokus vs SVD Nyström: p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
K-means vs SVD Nyström: p_raw=1.9375e-05, p_adj=4.0688e-04, reject H0=True

"""

# 90 epochs - k = 45, Kmeans = 45
"""
SVD k: {'method_1': 45, 'method_2': 45, 'method_3': 45, 'method_4': 45}
Hokus Pokus k: 45
K-means c: 45
SVD Nyström k: 45
method_1 (K): mean RMSE=2.394130e-01, std=3.058963e-02

method_2 (V): mean RMSE=1.807968e-01, std=1.755639e-02

method_3 (K & V sep): mean RMSE=2.815417e-01, std=2.944636e-02

method_4 (K & V joint): mean RMSE=3.079732e-01, std=3.248283e-02

Hokus Pokus: mean RMSE=1.537767e-01, std=1.715994e-02

K-means: mean RMSE=7.874794e-01, std=6.818255e-02

SVD Nyström: mean RMSE=1.186585e+00, std=9.018911e-01

method_1 (K) vs method_2 (V): p_raw=5.4981e-14, p_adj=1.1546e-12, reject H0=True
method_1 (K) vs method_3 (K & V sep): p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_1 (K) vs method_4 (K & V joint): p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_1 (K) vs Hokus Pokus: p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_1 (K) vs K-means: p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_1 (K) vs SVD Nyström: p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_2 (V) vs method_3 (K & V sep): p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_2 (V) vs method_4 (K & V joint): p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_2 (V) vs Hokus Pokus: p_raw=1.2743e-12, p_adj=2.6761e-11, reject H0=True
method_2 (V) vs K-means: p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_2 (V) vs SVD Nyström: p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_3 (K & V sep) vs method_4 (K & V joint): p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_3 (K & V sep) vs Hokus Pokus: p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_3 (K & V sep) vs K-means: p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_3 (K & V sep) vs SVD Nyström: p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_4 (K & V joint) vs Hokus Pokus: p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_4 (K & V joint) vs K-means: p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_4 (K & V joint) vs SVD Nyström: p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
Hokus Pokus vs K-means: p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
Hokus Pokus vs SVD Nyström: p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
K-means vs SVD Nyström: p_raw=1.6638e-12, p_adj=3.4939e-11, reject H0=True
"""


# 90 epochs - Hyperparameters from best performance
"""
SVD k: {'method_1': 62, 'method_2': 68, 'method_3': 68, 'method_4': 74}
Hokus Pokus k: 45
K-means c: 45
SVD Nyström k: 1
method_1 (K): mean RMSE=1.825364e-01, std=2.897773e-02

method_2 (V): mean RMSE=1.011896e-01, std=1.352263e-02

method_3 (K & V sep): mean RMSE=1.930992e-01, std=2.588209e-02

method_4 (K & V joint): mean RMSE=2.478220e-01, std=3.028214e-02

Hokus Pokus: mean RMSE=1.522743e-01, std=1.693692e-02

K-means: mean RMSE=7.767167e-01, std=6.432467e-02

SVD Nyström: mean RMSE=7.584465e-01, std=2.612349e-02

method_1 (K) vs method_2 (V): p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_1 (K) vs method_3 (K & V sep): p_raw=1.9889e-09, p_adj=4.1766e-08, reject H0=True
method_1 (K) vs method_4 (K & V joint): p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_1 (K) vs Hokus Pokus: p_raw=9.9906e-12, p_adj=2.0980e-10, reject H0=True
method_1 (K) vs K-means: p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_1 (K) vs SVD Nyström: p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_2 (V) vs method_3 (K & V sep): p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_2 (V) vs method_4 (K & V joint): p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_2 (V) vs Hokus Pokus: p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_2 (V) vs K-means: p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_2 (V) vs SVD Nyström: p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_3 (K & V sep) vs method_4 (K & V joint): p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_3 (K & V sep) vs Hokus Pokus: p_raw=1.0881e-13, p_adj=2.2850e-12, reject H0=True
method_3 (K & V sep) vs K-means: p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_3 (K & V sep) vs SVD Nyström: p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_4 (K & V joint) vs Hokus Pokus: p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_4 (K & V joint) vs K-means: p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
method_4 (K & V joint) vs SVD Nyström: p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
Hokus Pokus vs K-means: p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
Hokus Pokus vs SVD Nyström: p_raw=5.2804e-14, p_adj=1.1089e-12, reject H0=True
K-means vs SVD Nyström: p_raw=5.1160e-04, p_adj=1.0744e-02, reject H0=True
"""
