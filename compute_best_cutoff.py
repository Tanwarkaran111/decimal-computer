# compute_best_cutoff.py
# Reads benchmarks_compare_time.csv and recommends a karatsuba cutoff.
import pandas as pd
from pathlib import Path
import numpy as np

CSV = Path("benchmarks_compare_time.csv")
if not CSV.exists():
    raise SystemExit("benchmarks_compare_time.csv not found. Run benchmarks_compare_time.py first.")

df = pd.read_csv(CSV)

# Filter only karatsuba entries and focus on larger digit lengths where karatsuba matters
kar = df[df["algo"]=="karatsuba"].copy()
if kar.empty:
    raise SystemExit("No karatsuba rows found in CSV.")

# consider digit lengths >= 8 as the target region (change if you used different ranges)
target = kar[kar["digit_len"] >= 8]
if target.empty:
    target = kar  # fallback to all if no large-digit rows

# compute mean time per cutoff
group = target.groupby("cutoff")["time_sec"].mean().reset_index().sort_values("time_sec")
print("Mean time per cutoff (ascending):")
print(group)

best_cutoff = int(group.iloc[0]["cutoff"])
best_time = float(group.iloc[0]["time_sec"])
print(f"\nRecommended karatsuba cutoff (based on mean time for digit_len>=8): {best_cutoff} (mean time {best_time:.4g} s)")

# also print ratio vs schoolbook averaged over same (digit_len,size)
# build pivot of means
agg = df.groupby(["algo","cutoff","digit_len","size"])["time_sec"].mean().reset_index()
# pivot to compute ratio where both exist
pivot = agg.pivot_table(index=["digit_len","size"], columns=["algo","cutoff"], values="time_sec")
# compute average ratio across sizes/digits for each cutoff where both exist
ratios = {}
for c in sorted(kar["cutoff"].unique()):
    col = ("karatsuba", c)
    if col not in pivot.columns or ("schoolbook", "") not in pivot.columns:
        continue
    ratio = (pivot[col] / pivot[("schoolbook","")]).dropna().mean()
    ratios[c] = ratio
print("\nAverage time ratio karatsuba/schoolbook by cutoff (lower <1 means karatsuba faster):")
for c,r in sorted(ratios.items()):
    print(f" cutoff {c}: ratio {r:.3f}")

# Save recommended cutoff to file for other scripts to read
Path("recommended_karatsuba_cutoff.txt").write_text(str(best_cutoff))
print("\nSaved recommended cutoff to recommended_karatsuba_cutoff.txt")
