# analyze_results_v2.py
# Usage: python analyze_results_v2.py fl_results.csv
import sys, io, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from math import sqrt
from scipy import stats

def ci95(mean, std, n):
    # 95% CI width with t critical (n is seeds)
    if n <= 1: return (mean, np.nan, np.nan)
    t = stats.t.ppf(0.975, df=n-1)
    half = t * (std / sqrt(n))
    return (mean, mean - half, mean + half)

def paired_ttest(a, b):
    # arrays of replicate accuracies for two configs
    if len(a) != len(b) or len(a) < 2:
        return np.nan, np.nan
    t, p = stats.ttest_rel(a, b)
    return t, p

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_results_v2.py fl_results.csv")
        sys.exit(1)

    path = sys.argv[1]
    df = pd.read_csv(path)
    # ensure consistent types
    df["alpha"] = df["alpha"].astype(str)
    df["seeds"] = df["seeds"].fillna(2).astype(int)

    # ---------- TOP 12 ----------
    top = df.sort_values("acc_mean", ascending=False).head(12).reset_index(drop=True)
    top.to_csv("top_configs_v2.csv", index=False)

    # ---------- GROUP SUMMARIES + 95% CI ----------
    def summarize(group_cols):
        g = df.groupby(group_cols).agg(
            acc_mean=("acc_mean","mean"),
            acc_std=("acc_mean","std"),
            nrows=("acc_mean","count")
        ).reset_index()
        lows, highs = [], []
        for m, s, n in zip(g["acc_mean"], g["acc_std"].fillna(0), g["nrows"]):
            mean, lo, hi = ci95(m, s if s>0 else 1e-12, n)
            lows.append(lo); highs.append(hi)
        g["ci95_lo"] = lows; g["ci95_hi"] = highs
        return g.sort_values("acc_mean", ascending=False)

    by_algo   = summarize(["algo"])
    by_sigma  = summarize(["dp_sigma"])
    by_K      = summarize(["K"])
    by_frac   = summarize(["frac"])
    by_alpha  = summarize(["alpha"])
    by_tuple  = summarize(["algo","K","frac","alpha","dp_sigma"])

    by_algo.to_csv("summary_by_algo_v2.csv", index=False)
    by_sigma.to_csv("summary_by_sigma_v2.csv", index=False)
    by_K.to_csv("summary_by_K_v2.csv", index=False)
    by_frac.to_csv("summary_by_frac_v2.csv", index=False)
    by_alpha.to_csv("summary_by_alpha_v2.csv", index=False)
    by_tuple.to_csv("summary_full_tuples_v2.csv", index=False)

    # ---------- PICK A BASELINE & DO PAIRED TESTS ----------
    # Baseline = best mean among IID, σ in {0,0.01,0.05}, K in {10,20}, frac in {0.5,1.0}, algo in {fedavg,fedprox,fedavgm}
    cand = df[(df["alpha"]=="iid") &
              (df["dp_sigma"].isin([0.0,0.01,0.05])) &
              (df["K"].isin([10,20])) &
              (df["frac"].isin([0.5,1.0])) &
              (df["algo"].isin(["fedavg","fedprox","fedavgm"]))]

    best_row = cand.sort_values("acc_mean", ascending=False).iloc[0]
    baseline = {
        "algo": best_row["algo"], "K": int(best_row["K"]),
        "frac": float(best_row["frac"]), "alpha": "iid",
        "dp_sigma": float(best_row["dp_sigma"])
    }

    # Gather replicate accuracies for baseline
    def mask_eq(d, sel):
        m = np.ones(len(d), dtype=bool)
        for k,v in sel.items():
            m &= (d[k]==v)
        return m

    # Attempt to reconstruct per-seed points by matching rows with same tuple.
    # If your CSV already aggregates across seeds, we just compare means (no t-test).
    # We still output a "paired" file with p-values as NaN if not enough replicates.
    comp_rows = []
    for algo, K, frac, alpha, sigma in product(
        df["algo"].unique(),
        sorted(df["K"].unique()),
        sorted(df["frac"].unique()),
        sorted(df["alpha"].astype(str).unique()),
        sorted(df["dp_sigma"].unique()),
    ):
        sel = {"algo":algo,"K":K,"frac":frac,"alpha":str(alpha),"dp_sigma":sigma}
        a_mean = df[mask_eq(df, sel)]["acc_mean"]
        b_mean = df[mask_eq(df, baseline)]["acc_mean"]
        # Simulate replicate arrays (if only means exist, t-test can’t be done)
        a_repl = a_mean.values
        b_repl = b_mean.values
        t, p = paired_ttest(a_repl, b_repl)
        comp_rows.append({**sel,
                          "acc_mean": a_mean.mean() if len(a_mean) else np.nan,
                          "vs_baseline_algo": baseline["algo"],
                          "vs_baseline_K": baseline["K"],
                          "vs_baseline_frac": baseline["frac"],
                          "vs_baseline_alpha": baseline["alpha"],
                          "vs_baseline_sigma": baseline["dp_sigma"],
                          "baseline_mean": b_mean.mean() if len(b_mean) else np.nan,
                          "t_stat": t, "p_value": p})

    comp = pd.DataFrame(comp_rows).dropna(subset=["acc_mean"]).sort_values("acc_mean", ascending=False)
    comp.to_csv("compare_vs_baseline_v2.csv", index=False)

    # ---------- PLOTS ----------
    os.makedirs("figs_v2", exist_ok=True)

    # 1) Acc vs σ for K=10, frac=0.5 under IID
    subset = df[(df["K"]==10) & (df["frac"]==0.5) & (df["alpha"]=="iid")]
    plt.figure()
    for algo in sorted(subset["algo"].unique()):
        s = subset[subset["algo"]==algo].sort_values("dp_sigma")
        plt.plot(s["dp_sigma"], s["acc_mean"], marker="o", label=algo)
    plt.title("Accuracy vs DP noise σ (K=10, frac=0.5, IID)")
    plt.xlabel("σ")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs_v2/acc_vs_sigma_K10_frac05_IID.png")
    plt.close()

    # 2) Acc vs K at σ=0.01 under IID, frac=1.0
    subset2 = df[(df["frac"]==1.0) & (df["alpha"]=="iid") & (df["dp_sigma"]==0.01)]
    plt.figure()
    for algo in sorted(subset2["algo"].unique()):
        s = subset2[subset2["algo"]==algo].sort_values("K")
        plt.plot(s["K"], s["acc_mean"], marker="o", label=algo)
    plt.title("Accuracy vs #clients K (frac=1.0, IID, σ=0.01)")
    plt.xlabel("K")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs_v2/acc_vs_K_frac10_IID_sigma001.png")
    plt.close()

    # 3) IID vs non-IID for K=20, frac=1.0, σ=0.01
    subset3 = df[(df["K"]==20) & (df["frac"]==1.0) & (df["dp_sigma"]==0.01)]
    alpha_map = {"iid":0, "0.3":1}
    plt.figure()
    for algo in sorted(subset3["algo"].unique()):
        s = subset3[subset3["algo"]==algo].copy()
        s["alpha_num"] = s["alpha"].map(alpha_map)
        s = s.sort_values("alpha_num")
        plt.plot(s["alpha_num"], s["acc_mean"], marker="o", label=algo)
    plt.title("IID vs non-IID (K=20, frac=1.0, σ=0.01)")
    plt.xlabel("Data Heterogeneity (0=IID, 1=α0.3)")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs_v2/IID_vs_nonIID_K20_frac10_sigma001.png")
    plt.close()

    print("\nSaved:")
    print(" - top_configs_v2.csv")
    print(" - summary_by_*_v2.csv")
    print(" - compare_vs_baseline_v2.csv (includes p-values where possible)")
    print(" - figs_v2/*.png")

if __name__ == "__main__":
    main()
