# phase3/generate_report.py
"""
Generate a short Markdown report summarizing Phase3 results.
"""
import pandas as pd, os, datetime
AGG = "phase3_benchmarks_aggregated.csv"
SELECTOR = os.path.join("phase3_results","phase3_selector_table.csv")
OUT = os.path.join("phase3_results","phase3_report.md")

def main():
    os.makedirs("phase3_results", exist_ok=True)
    df = pd.read_csv(AGG)
    sel = pd.read_csv(SELECTOR) if os.path.exists(SELECTOR) else None

    now = datetime.datetime.now().isoformat()
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(f"# Phase 3 Performance Report\n\nGenerated: {now}\n\n")
        f.write("## Overview\n\nThis report summarizes benchmark results (mean runtime + stdev) and the auto-selector table.\n\n")
        f.write("## Per-algo summary (sample)\n\n")
        for algo in sorted(df['algo'].unique()):
            sub = df[df['algo']==algo]
            best = sub.sort_values('mean_seconds').head(1).iloc[0]
            f.write(f"- **{algo}** fastest sample: n={int(best['n'])} digits={int(best['digits'])} mean={best['mean_seconds']:.6f}s\n")
        f.write("\n## Selector table (best algos)\n\n")
        if sel is not None:
            f.write(sel.to_markdown(index=False))
            f.write("\n\n")
        f.write("## Plots\n\nInclude generated plot files from `phase3_results/` (mean_vs_size, bar charts).\n\n")
    print("Wrote", OUT)

if __name__ == "__main__":
    main()
