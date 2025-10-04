# plot_benchmarks.py
# Read benchmarks.csv (created by benchmarks.py), plot results and fit a simple model.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression

CSV = Path("benchmarks.csv")
if not CSV.exists():
    raise SystemExit("benchmarks.csv not found in current folder. Run benchmarks.py first.")

df = pd.read_csv(CSV)
# Aggregate mean across trials
agg = df.groupby(["digit_len","size"]).agg({"muls":"mean","adds":"mean"}).reset_index()

# Plot multiplications and additions (log-log view)
fig, axes = plt.subplots(1,2, figsize=(14,5))
for d in sorted(agg["digit_len"].unique()):
    sub = agg[agg["digit_len"]==d]
    axes[0].plot(sub["size"], sub["muls"], marker='o', label=f"{d} digits")
    axes[1].plot(sub["size"], sub["adds"], marker='o', label=f"{d} digits")

axes[0].set_xscale('log', base=2)
axes[1].set_xscale('log', base=2)
axes[0].set_yscale('log')
axes[1].set_yscale('log')

axes[0].set_title("Mean digit multiplications vs matrix size")
axes[1].set_title("Mean digit additions vs matrix size")
axes[0].set_xlabel("matrix size n (log scale)")
axes[1].set_xlabel("matrix size n (log scale)")
axes[0].set_ylabel("mean digit multiplications")
axes[1].set_ylabel("mean digit additions")

axes[0].legend()
axes[1].legend()
axes[0].grid(True, which='both', ls='--', alpha=0.4)
axes[1].grid(True, which='both', ls='--', alpha=0.4)

plt.tight_layout()
out_png = "benchmarks_plots.png"
plt.savefig(out_png, dpi=200)
print(f"Saved plot to {out_png}")
plt.show()

# Fit model muls ≈ c * n^3 * d^2  (linearize by taking logs)
df_fit = agg.copy()
df_fit["X"] = 3 * np.log(df_fit["size"]) + 2 * np.log(df_fit["digit_len"])
df_fit["Y"] = np.log(df_fit["muls"])

X = df_fit[["X"]].to_numpy()
Y = df_fit["Y"].to_numpy().reshape(-1,1)

lr = LinearRegression().fit(X, Y)
slope = lr.coef_[0,0]
intercept = lr.intercept_[0]
c = float(np.exp(intercept))

print("\nFit result for model: muls ≈ c * n^3 * d^2")
print(f"  slope (should be ~1 if exponents 3 and 2 are correct): {slope:.4f}")
print(f"  c (constant factor): {c:.4f}")

# show sample table comparing measured vs predicted
df_fit["pred_muls"] = c * (df_fit["size"]**3) * (df_fit["digit_len"]**2)
df_fit["ratio"] = df_fit["muls"] / df_fit["pred_muls"]
print("\nSample rows (digit_len, size, mean_muls, pred_muls, ratio):")
print(df_fit[["digit_len","size","muls","pred_muls","ratio"]].round(3).head(12))
