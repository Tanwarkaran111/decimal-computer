import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys

# === CONFIG ===
CSV_PATH = Path("D:/Invented_Library/experiments/bench_strategies.csv")
OUT_JSON = Path("D:/Invented_Library/recommended_cutoffs.json")

if not CSV_PATH.exists():
    print(f"❌ Error: {CSV_PATH} not found")
    sys.exit(1)

# === LOAD & CLEAN ===
df = pd.read_csv(CSV_PATH, on_bad_lines="skip")
df = df.dropna(subset=["size", "digits", "algo", "elapsed"])

df["size"] = df["size"].astype(int)
df["digits"] = df["digits"].astype(int)
df["elapsed"] = df["elapsed"].astype(float)

# === SUMMARY ===
print("\n=== Benchmark Summary ===")
avg_df = df.groupby(["size", "digits", "algo"], as_index=False)["elapsed"].mean()

best = avg_df.loc[avg_df.groupby(["size", "digits"])["elapsed"].idxmin()]
for _, row in best.iterrows():
    print(f"n={row['size']}, d={row['digits']} -> best={row['algo']} ({row['elapsed']:.6f}s)")

# === CROSSOVER DETECTION ===
print("\n=== Crossover Detection ===")
cutoffs = {}

for digits in sorted(df["digits"].unique()):
    subset = avg_df[avg_df["digits"] == digits]
    if subset.empty:
        continue

    pivot = subset.pivot(index="size", columns="algo", values="elapsed")
    algos = list(pivot.columns)

    cutoffs[str(digits)] = {"karatsuba": None, "strassen": None}

    if "schoolbook" in algos and "karatsuba" in algos:
        for size in sorted(pivot.index):
            if pivot.loc[size, "karatsuba"] < pivot.loc[size, "schoolbook"]:
                print(f"Karatsuba beats Schoolbook starting at n={size}, d={digits}")
                cutoffs[str(digits)]["karatsuba"] = int(size)
                break

    if "schoolbook" in algos and "strassen" in algos:
        for size in sorted(pivot.index):
            if pivot.loc[size, "strassen"] < pivot.loc[size, "schoolbook"]:
                print(f"Strassen beats Schoolbook starting at n={size}, d={digits}")
                cutoffs[str(digits)]["strassen"] = int(size)
                break

# === EXPORT CUTOFFS JSON ===
with open(OUT_JSON, "w") as f:
    json.dump(cutoffs, f, indent=2)

print(f"\n✅ Exported cutoffs to {OUT_JSON}")

# === PLOTTING ===
for digits in sorted(df["digits"].unique()):
    subset = avg_df[avg_df["digits"] == digits]
    if subset.empty:
        continue

    plt.figure(figsize=(8, 6))
    for algo in subset["algo"].unique():
        algo_data = subset[subset["algo"] == algo]
        plt.plot(algo_data["size"], algo_data["elapsed"], marker="o", label=algo)

    plt.title(f"Algo Performance (digits={digits})")
    plt.xlabel("Matrix size (n)")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    outpath = f"benchmark_plot_digits{digits}.png"
    plt.savefig(outpath)
    print(f"📊 Saved plot: {outpath}")

print("\nAnalysis complete. Open the PNG plots and JSON cutoffs.")
