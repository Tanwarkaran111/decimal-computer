# plot_compare.py
# Read benchmarks_compare.csv and plot comparisons between schoolbook and karatsuba.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression

CSV = Path("benchmarks_compare.csv")
if not CSV.exists():
    raise SystemExit("benchmarks_compare.csv not found. Run benchmarks_compare.py first.")

df = pd.read_csv(CSV)

# quick sanity
print("Rows:", len(df))
print(df.head())

# Aggregate mean across trials for plotting
agg = df.groupby(["algo","digit_len","size"]).agg({"muls":"mean","adds":"mean"}).reset_index()

# Plotting: for each digit_len, show both algos across sizes
digit_lengths = sorted(agg["digit_len"].unique())
algos = sorted(agg["algo"].unique())
colors = {"schoolbook":"tab:blue","karatsuba":"tab:orange"}

fig, axes = plt.subplots(1,2, figsize=(14,5))
for d in digit_lengths:
    for algo in algos:
        sub = agg[(agg["digit_len"]==d) & (agg["algo"]==algo)]
        axes[0].plot(sub["size"], sub["muls"], marker='o', label=f"{algo} ({d}d)", color=colors[algo], linestyle='-' if algo=="schoolbook" else '--')
        axes[1].plot(sub["size"], sub["adds"], marker='o', label=f"{algo} ({d}d)", color=colors[algo], linestyle='-' if algo=="schoolbook" else '--')

for ax in axes:
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, which='both', ls='--', alpha=0.4)

axes[0].set_title("Mean digit multiplications vs matrix size (compare algos)")
axes[1].set_title("Mean digit additions vs matrix size (compare algos)")
axes[0].set_xlabel("matrix size n (log scale)")
axes[1].set_xlabel("matrix size n (log scale)")
axes[0].set_ylabel("mean digit multiplications")
axes[1].set_ylabel("mean digit additions")
axes[0].legend()
axes[1].legend()
plt.tight_layout()
out_png = "benchmarks_compare_plots.png"
plt.savefig(out_png, dpi=200)
print("Saved plot to", out_png)
plt.show()

# ----------------------
# Fit a model: muls ≈ C * n^a * d^b  (so log(muls) = log C + a*log n + b*log d)
# Fit separately for each algorithm and print exponents a and b.
# ----------------------
df_fit = agg.copy()
df_fit["ln_n"] = np.log(df_fit["size"].astype(float))
df_fit["ln_d"] = np.log(df_fit["digit_len"].astype(float))
df_fit["ln_muls"] = np.log(df_fit["muls"].astype(float))

print("\nFitting muls ≈ C * n^a * d^b for each algorithm:")
for algo in algos:
    sub = df_fit[df_fit["algo"]==algo]
    X = sub[["ln_n","ln_d"]].to_numpy()
    Y = sub["ln_muls"].to_numpy().reshape(-1,1)
    lr = LinearRegression().fit(X, Y)
    a = float(lr.coef_[0,0])
    b = float(lr.coef_[0,1])
    intercept = float(lr.intercept_[0])
    C = float(np.exp(intercept))
    print(f"Algorithm: {algo}")
    print(f"  fitted a (n-exponent) = {a:.4f}")
    print(f"  fitted b (d-exponent) = {b:.4f}")
    print(f"  fitted C (constant)   = {C:.4g}")
    print("  (Expected naive: a≈3, b≈2 ; Karatsuba should show smaller b when it helps)\n")

# Also print average ratio of karatsuba/schoolbook muls per (d,size)
pivot = agg.pivot_table(index=["digit_len","size"], columns="algo", values="muls").reset_index()
pivot["ratio_k_over_s"] = pivot["karatsuba"] / pivot["schoolbook"]
print("Sample pivot (digit_len,size, schoolbook_muls, karatsuba_muls, ratio_karatsuba/schoolbook):")
print(pivot.head(12).round(3))
