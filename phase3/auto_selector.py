# phase3/auto_selector.py
"""
Generate an auto-selector table from aggregated benchmark CSV.
Produces:
 - phase3_selector_table.csv   (mapping n,digits -> best_algo,mean,stdev)
 - phase3_selector.py          (simple runtime chooser function)

Usage:
  python -m phase3.auto_selector --agg-file ./phase3_benchmarks_aggregated.csv --out-dir ./phase3_results
"""
import argparse
import os
import pandas as pd

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
    df['digits'] = df['digits'].astype(int)
    df['mean_seconds'] = df['mean_seconds'].astype(float)
    # pick best algo per (n,digits)
    best = df.loc[df.groupby(['n','digits'])['mean_seconds'].idxmin()].reset_index(drop=True)
    out_csv = os.path.join(args.out_dir, "phase3_selector_table.csv")
    best.to_csv(out_csv, index=False)
    print("Wrote", out_csv)

    # write a tiny python selector module
    sel_py = os.path.join(args.out_dir, "phase3_selector.py")
    with open(sel_py, "w", encoding="utf-8") as fh:
        fh.write("# Auto-generated selector: choose_algo(n,digits)\n")
        fh.write("selector_table = {\n")
        for _, row in best.iterrows():
            fh.write(f"    ({int(row['n'])},{int(row['digits'])}): '{row['algo']}',\n")
        fh.write("}\n\n")
        fh.write("def choose_algo(n, digits):\n")
        fh.write("    # Exact-match selector. If no exact match, falls back to nearest smaller n with same digits.\n")
        fh.write("    key = (int(n), int(digits))\n")
        fh.write("    if key in selector_table:\n")
        fh.write("        return selector_table[key]\n")
        fh.write("    # fallback search: same digits, largest n <= requested\n")
        fh.write("    candidates = [(k,v) for k,v in selector_table.items() if k[1]==int(digits) and k[0] <= int(n)]\n")
        fh.write("    if candidates:\n")
        fh.write("        best_key = max(candidates, key=lambda kv: kv[0][0])[0]\n")
        fh.write("        return selector_table[best_key]\n")
        fh.write("    # absolute fallback\n")
        fh.write("    return 'schoolbook'\n")
    print("Wrote", sel_py)
    print("Done.")

if __name__ == '__main__':
    main()
