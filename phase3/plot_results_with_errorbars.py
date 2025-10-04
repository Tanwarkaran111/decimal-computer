# phase3/plot_results_with_errorbars.py
"""
Plot aggregated results with error bars (mean ± stdev) without using NumPy.

Usage:
  python -m phase3.plot_results_with_errorbars --agg-file ./phase3_benchmarks_aggregated.csv --out-dir ./phase3_results
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agg-file", required=True)
    p.add_argument("--out-dir", default=".")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.agg_file)
    df['n'] = df['n'].astype(int)
    df['mean_seconds'] = df['mean_seconds'].astype(float)
    df['stdev_seconds'] = df['stdev_seconds'].astype(float)

    digits_values = sorted(df['digits'].unique())
    for digits in digits_values:
        sub = df[df['digits'] == digits]
        plt.figure(figsize=(8,5))
        for algo in sorted(sub['algo'].unique()):
            s = sub[sub['algo'] == algo].sort_values('n')
            if s.empty:
                continue
            x = s['n'].tolist()
            y = s['mean_seconds'].tolist()
            e = s['stdev_seconds'].tolist()
            plt.errorbar(x, y, yerr=e, marker='o', capsize=4, label=algo)
        plt.xlabel("matrix size (n)")
        plt.ylabel("mean runtime (s)")
        plt.title(f"Mean runtime vs size — digits={digits}")
        plt.legend()
        out = os.path.join(args.out_dir, f"phase3_mean_vs_size_d{digits}_err.png")
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        print("Wrote", out)

    # bar chart at largest n with error bars
    max_n = int(df['n'].max())
    top = df[df['n'] == max_n].sort_values(['digits','algo'])
    labels = []
    heights = []
    errors = []
    for _, row in top.iterrows():
        labels.append(f"{row['algo']}\nd={int(row['digits'])}")
        heights.append(float(row['mean_seconds']))
        errors.append(float(row['stdev_seconds']))

    plt.figure(figsize=(10,5))
    x_positions = list(range(len(heights)))
    plt.bar(x_positions, heights, yerr=errors, capsize=5)
    plt.xticks(x_positions, labels, rotation=45, ha='right')
    plt.ylabel("mean runtime (s)")
    plt.title(f"Comparison at n={max_n}")
    out2 = os.path.join(args.out_dir, f"phase3_bar_n{max_n}_err.png")
    plt.tight_layout()
    plt.savefig(out2)
    plt.close()
    print("Wrote", out2)

if __name__ == "__main__":
    main()
