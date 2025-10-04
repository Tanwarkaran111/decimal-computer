# experiments/plot_compare_time.py
import pandas as pd
import matplotlib.pyplot as plt
import argparse

p = argparse.ArgumentParser()
p.add_argument("--input", required=True, help="CSV file (long format)")
p.add_argument("--out", default="time_comparison.png", help="Output plot file")
args = p.parse_args()

df = pd.read_csv(args.input)

# Simple plot: compare mean time vs size (n), grouped by algo (numpy vs decimal)
plt.figure(figsize=(8,6))
for algo, g in df.groupby("algo"):
    plt.plot(g["n"], g["mean_s"], marker="o", label=algo)

plt.xlabel("Matrix size (n)")
plt.ylabel("Mean runtime (s)")
plt.title("NumPy vs Decimal runtime")
plt.legend()
plt.grid(True)
plt.savefig(args.out, dpi=150)
print(f"Saved plot -> {args.out}")
