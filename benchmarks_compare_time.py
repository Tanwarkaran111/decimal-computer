# benchmarks_compare_time.py
# Add per-trial timing (time_s) and produce aggregated CSV with mean/std for time, muls, adds.
import csv
import time
from pathlib import Path
from statistics import mean, stdev
from decimal_computer.decimal_gemm import decimal_gemm_naive
# --- add near top of benchmarks_compare_time.py ---
import csv
from datetime import datetime
from pathlib import Path

TIMING_CSV = Path("benchmarks_time_log.csv")

def _ensure_csv_header():
    if not TIMING_CSV.exists():
        with TIMING_CSV.open("w", newline="") as fh:
            writer = csv.writer(fh)
            header = ["ts","experiment","digit_len","size","trial","algo","cutoff","muls","adds","time_s"]
            writer.writerow(header)

def log_timing(experiment, digit_len, size, trial, algo, cutoff, muls, adds, time_s):
    """Append a single detailed timing row to benchmarks_time_log.csv"""
    _ensure_csv_header()
    row = [
        datetime.utcnow().isoformat() + "Z",
        experiment,
        int(digit_len),
        int(size),
        int(trial),
        str(algo),
        "" if cutoff is None else int(cutoff),
        int(muls) if muls is not None else "",
        int(adds) if adds is not None else "",
        float(time_s)
    ]
    with TIMING_CSV.open("a", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(row)

# ... then inside the loop where you run a single trial (example) ...
# after you get muls, adds, elapsed_time:
# log_timing("compare_time", d, size, trial_id, algo_name, cutoff, muls, adds, elapsed_time)

OUT_CSV = Path("benchmarks_compare_time.csv")
OUT_AGG_CSV = Path("benchmarks_compare_time_agg.csv")

def run_compare_time(matrix_sizes=(4,8,16), digit_lengths=(4,8,16,32), trials=3, karatsuba_cutoffs=(8,16,32,64)):
    """
    For each (digit_len, matrix_size, trial, algo, cutoff) run decimal_gemm_naive
    and record: muls, adds, time_s.
    """
    rows = []
    algos = ["schoolbook", "karatsuba", "auto"]
    for d in digit_lengths:
        for n in matrix_sizes:
            for trial in range(trials):
                for algo in algos:
                    # decide cutoffs to try
                    cutoffs = [None]
                    if algo == "karatsuba":
                        cutoffs = list(karatsuba_cutoffs)
                    # if algo is auto we still want to record once with cutoff=None so auto picks
                    if algo == "auto":
                        cutoffs = [None]
                    for cutoff in cutoffs:
                        # Build example matrices: use small ints to keep time reasonable
                        # You can replace this with your existing generator if you have one.
                        A = [[(i + j + d) % 100 for j in range(n)] for i in range(n)]
                        B = [[(i * 2 + j + d) % 100 for j in range(n)] for i in range(n)]

                        # Run and time
                        t0 = time.perf_counter()
                        # decimal_gemm_naive returns C or (C, muls, adds) depending flags.
                        # Use return_counters=True to get muls/adds directly.
                        try:
                            result = decimal_gemm_naive(A, B, return_counters=True, mul_algo=algo if algo != "auto" else "auto", karatsuba_cutoff=cutoff)
                        except TypeError:
                            # some versions of your function might expect different kwarg names; call flexibly
                            result = decimal_gemm_naive(A, B, return_counters=True, mul_algo=algo if algo != "auto" else "auto", cutoff=cutoff)
                        t1 = time.perf_counter()
                        # result may be (C, muls, adds) or (C, muls, adds, elapsed)
                        if len(result) >= 3:
                            C, muls, adds = result[0], result[1], result[2]
                        else:
                            C = result
                            # fallback: if counters not returned, run with return_counters=True explicitly above
                            muls, adds = None, None

                        elapsed = t1 - t0
                        rows.append({
                            "digit_len": d,
                            "size": n,
                            "trial": trial,
                            "algo": algo,
                            "cutoff": "" if cutoff is None else cutoff,
                            "muls": muls,
                            "adds": adds,
                            "time_s": elapsed
                        })
                        print(f"[d={d} n={n} t={trial} algo={algo} cutoff={cutoff}] muls={muls} adds={adds} time_s={elapsed:.6f}")

    # write raw CSV
    keys = ["digit_len", "size", "trial", "algo", "cutoff", "muls", "adds", "time_s"]
    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, keys)
        writer.writeheader()
        writer.writerows(rows)
    print("Saved raw per-trial results to", OUT_CSV)

    # aggregate: mean & std for muls/adds/time grouped by (digit_len,size,algo,cutoff)
    groups = {}
    for r in rows:
        key = (r["digit_len"], r["size"], r["algo"], r["cutoff"])
        groups.setdefault(key, {"muls": [], "adds": [], "time_s": []})
        if r["muls"] is not None:
            groups[key]["muls"].append(r["muls"])
        if r["adds"] is not None:
            groups[key]["adds"].append(r["adds"])
        groups[key]["time_s"].append(r["time_s"])

    agg_rows = []
    for (d,n,algo,cutoff), vals in groups.items():
        def mean_std(l):
            if not l:
                return (None, None)
            if len(l) == 1:
                return (mean(l), 0.0)
            return (mean(l), stdev(l))
        mul_mean, mul_std = mean_std(vals["muls"])
        add_mean, add_std = mean_std(vals["adds"])
        time_mean, time_std = mean_std(vals["time_s"])
        agg_rows.append({
            "digit_len": d,
            "size": n,
            "algo": algo,
            "cutoff": cutoff,
            "muls_mean": mul_mean,
            "muls_std": mul_std,
            "adds_mean": add_mean,
            "adds_std": add_std,
            "time_mean": time_mean,
            "time_std": time_std
        })

    # write aggregated CSV
    agg_keys = ["digit_len","size","algo","cutoff","muls_mean","muls_std","adds_mean","adds_std","time_mean","time_std"]
    with OUT_AGG_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, agg_keys)
        w.writeheader()
        w.writerows(agg_rows)
    print("Saved aggregated results to", OUT_AGG_CSV)

    # print a small sample
    print("Sample aggregated rows (first 8):")
    for r in agg_rows[:8]:
        print(r)

if __name__ == "__main__":
    # default parameters are modest; increase if you want more data but it will take longer.
    run_compare_time(matrix_sizes=(2,4,8), digit_lengths=(1,2,4,8), trials=3, karatsuba_cutoffs=(8,16,32))
