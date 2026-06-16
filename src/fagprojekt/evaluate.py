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
    best_k_sweep = {"method_1": [37, 45, 63], "method_2": [22, 45, 75], "method_3": [25, 45, 69], "method_4": [22, 45, 63]}
    get_rmse_companies_SVD(model=model, tokenizer=tokenizer, layer_idx=layer_idx, head_idx=head_idx, num_tokens=num_tokens, thresholds_per_method=best_k_sweep, companies=test_companies, path_suffix="_test", want_plot=False)
    df = pd.read_csv(f"reports/figures/SVD/k_tuning_all_layer_{layer_idx}_head_{head_idx}_tokens_{num_tokens}_test.csv")

    best_k_svd = {"method_1": 63, "method_2": 75, "method_3": 69, "method_4": 63}
    rmse_per_SVD_method = {
        method: df[(df["method"] == method) & (df["k"] == k)]
                .sort_values(["company", "page"])["rmse"].values
        for method, k in best_k_svd.items()
    }

    # ------ Hokus Pokus ------
    k_hp = 45
    rmse_Hokus_pokus = get_rmse_companies_Hokus_Pokus(model=model, tokenizer=tokenizer, num_tokens=num_tokens, layer_idx=layer_idx, head_idx=head_idx, k=k_hp, train_companies=train_companies, test_companies=test_companies, num_epochs=num_epochs)

    # ------ K-means ------
    clusters = 8
    rmse_K_means = get_rmse_companies_K_means(model=model, tokenizer=tokenizer, layer_idx=layer_idx, head_idx=head_idx, num_tokens=num_tokens, clusters=clusters, companies=test_companies)

    # ------ Statistical tests ------
    all_rmse = {
        "method_1": rmse_per_SVD_method["method_1"],
        "method_2": rmse_per_SVD_method["method_2"],
        "method_3": rmse_per_SVD_method["method_3"],
        "method_4": rmse_per_SVD_method["method_4"],
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
    num_epochs = int(os.environ["NUM_EPOCHS"])
    
    model, tokenizer = load_model(want_print=False)

    train_companies = sorted(['GoldmanSachs', 'HSBC', 'JPMorgan', 'Kroger', 'NewRiver', 'PNC', 'Reach', 'Sagicor', 'United', 'UPS', 'Vesuvius', 'WoltersKluwer'])
    test_companies  = sorted(['AIG', 'AmericanAirlines', 'APA'])

    run_evaluation_rmse(model, tokenizer, layer_idx, head_idx, num_tokens, train_companies, test_companies, num_epochs)

    torch.cuda.syncronize()
    prof.step()
    prof.stop()

    with open("eval_metrics.txt", "a") as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    # First try - 1 Pass over training data to train g (1 epoch)
    """
    method_1: mean RMSE=1.909066e-02, std=2.825224e-03
    method_2: mean RMSE=1.815607e-02, std=2.605675e-03
    method_3: mean RMSE=2.615147e-02, std=3.296686e-03
    method_4: mean RMSE=1.353081e-02, std=1.629130e-03
    Hokus Pokus: mean RMSE=7.843776e-01, std=1.110772e-01
    K-means: mean RMSE=8.498689e-02, std=9.139702e-03

    method_1 vs method_2: p_raw=3.3698e-06, p_adj=5.0547e-05, reject H0=True
    method_1 vs method_3: p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
    method_1 vs method_4: p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
    method_1 vs Hokus Pokus: p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
    method_1 vs K-means: p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
    method_2 vs method_3: p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
    method_2 vs method_4: p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
    method_2 vs Hokus Pokus: p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
    method_2 vs K-means: p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
    method_3 vs method_4: p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
    method_3 vs Hokus Pokus: p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
    method_3 vs K-means: p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
    method_4 vs Hokus Pokus: p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
    method_4 vs K-means: p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
    Hokus Pokus vs K-means: p_raw=5.2804e-14, p_adj=7.9206e-13, reject H0=True
    """