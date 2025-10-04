# plot_compare_time.py  (robustified)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression

CSV = Path("benchmarks_compare_time.csv")
if not CSV.exists():
    raise SystemExit("benchmarks_compare_time.csv not found. Run benchmarks_compare_time.py first.")

df = pd.read_csv(CSV)

# aggregate mean across trials
agg = df.groupby(["algo","cutoff","digit_len","size"]).agg({
    "muls":"mean","adds":"mean","time_sec":"mean"
}).reset_index()

if agg.shape[0] == 0:
    raise SystemExit("Aggregated data is empty. Check benchmarks_compare_time.csv contents.")

# Plot muls and time (separate)
digit_lengths = sorted(agg["digit_len"].unique())
algos = sorted(agg["algo"].unique())

fig, axes = plt.subplots(1,2, figsize=(14,5))
for d in digit_lengths:
    for algo in algos:
        if algo == "schoolbook":
            sub = agg[(agg["algo"]==algo) & (agg["digit_len"]==d)]
            if sub.empty:
                continue
            axes[0].plot(sub["size"], sub["muls"], marker='o', label=f"{algo} ({d}d)")
            axes[1].plot(sub["size"], sub["time_sec"], marker='o', label=f"{algo} ({d}d)")
        else:
            for cutoff in sorted(agg["cutoff"].unique()):
                sub = agg[(agg["algo"]=="karatsuba") & (agg["digit_len"]==d) & (agg["cutoff"]==cutoff)]
                if sub.empty:
                    continue
                axes[0].plot(sub["size"], sub["muls"], marker='o', linestyle='--', label=f"karatsuba c={cutoff} ({d}d)")
                axes[1].plot(sub["size"], sub["time_sec"], marker='o', linestyle='--', label=f"karatsuba c={cutoff} ({d}d)")

for ax in axes:
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, which='both', ls='--', alpha=0.4)

axes[0].set_title("Mean digit multiplications vs matrix size (compare algos & cutoffs)")
axes[1].set_title("Mean wall-clock time vs matrix size (compare algos & cutoffs)")
axes[0].set_xlabel("matrix size n (log scale)")
axes[1].set_xlabel("matrix size n (log scale)")
axes[0].set_ylabel("mean digit multiplications")
axes[1].set_ylabel("mean time (sec)")
axes[0].legend(fontsize="small", ncol=2)
axes[1].legend(fontsize="small", ncol=2)
plt.tight_layout()
out_png = "benchmarks_compare_time_plots.png"
plt.savefig(out_png, dpi=200)
print("Saved plot to", out_png)
plt.show()

# ----------------------
# Fit a model: muls ≈ C * n^a * d^b for schoolbook and karatsuba (aggregating karatsuba cutoffs)
# ----------------------
print("\nFitting muls ≈ C * n^a * d^b for each algorithm (aggregating karatsuba cutoffs):")
df_fit = agg.copy()
df_fit["ln_n"] = np.log(df_fit["size"].astype(float))
df_fit["ln_d"] = np.log(df_fit["digit_len"].astype(float))
df_fit["ln_muls"] = np.log(df_fit["muls"].astype(float))

for algo in ["schoolbook","karatsuba"]:
    sub = df_fit[df_fit["algo"]==algo]
    if sub.empty:
        print(f"Skipping fit for '{algo}': no data available.")
        continue
    X = sub[["ln_n","ln_d"]].to_numpy()
    Y = sub["ln_muls"].to_numpy().reshape(-1,1)
    # Extra safety: require at least 2 rows to fit both coefficients
    if X.shape[0] < 2:
        print(f"Skipping fit for '{algo}': not enough rows ({X.shape[0]}).")
        continue
    try:
        lr = LinearRegression().fit(X, Y)
    except Exception as e:
        print(f"Fit failed for '{algo}': {e}")
        continue
    a = float(lr.coef_[0,0])
    b = float(lr.coef_[0,1])
    intercept = float(lr.intercept_[0])
    C = float(np.exp(intercept))
    print(f"\nAlgorithm: {algo}")
    print(f"  fitted a (n-exponent) = {a:.4f}")
    print(f"  fitted b (d-exponent) = {b:.4f}")
    print(f"  fitted C (constant)   = {C:.4g}")
    print()
