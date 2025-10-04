# phase2/tuner.py
"""
Tuner: benchmark the three backends and suggest cutoffs.
Produces:
 - phase2/cutoffs.json  (updated)
 - phase2/tuner_plot_d4.png, tuner_plot_d8.png, tuner_plot_d16.png
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, List, Tuple

# Import matrix generators and backends from hybrid_core (local)
from phase2.hybrid_core import (
    schoolbook_gemm,
    karatsuba_gemm,
    strassen_gemm,
    random_matrix,
)

OUT_DIR = Path(__file__).parent
CUTOFF_FILE = OUT_DIR / "cutoffs.json"

def bench(func: Callable, n: int, digits: int, trials: int = 3) -> float:
    """Run func(A,B,digits=digits) for `trials` and return mean elapsed time."""
    times = []
    for _ in range(trials):
        A = random_matrix(n, n, digits)
        B = random_matrix(n, n, digits)
        t0 = time.perf_counter()
        # different backends expect either (A,B) or (A,B,cutoff)
        try:
            func(A, B)
        except TypeError:
            # fallback if backend requires cutoff or other kw
            func(A, B, cutoff=None)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return mean(times)

def tune(sizes: List[int] = None, digits_list: List[int] = None):
    sizes = sizes or [8, 16, 32, 64, 128, 256]
    digits_list = digits_list or [4, 8, 16]

    results: Dict[int, Dict[str, List[Tuple[int, float]]]] = {}
    cutoffs: Dict[str, int] = {}

    print(">> Running tuner...")

    for d in digits_list:
        results[d] = {"school": [], "karatsuba": [], "strassen": []}
        last_algo = "schoolbook"
        # for detecting cross-over we keep last times
        for n in sizes:
            t_school = bench(schoolbook_gemm, n, d, trials=3)
            t_karat = bench(karatsuba_gemm, n, d, trials=3)
            t_stras = bench(strassen_gemm, n, d, trials=2)
            results[d]["school"].append((n, t_school))
            results[d]["karatsuba"].append((n, t_karat))
            results[d]["strassen"].append((n, t_stras))

            print(f"[tune] n={n}, d={d} -> school={t_school:.6f}s karatsuba={t_karat:.6f}s strassen={t_stras:.6f}s")

            # detect schoolbook -> karatsuba cutoff
            key1 = f"schoolbook_vs_karatsuba_d{d}"
            if last_algo == "schoolbook" and t_karat < t_school:
                # first n where karatsuba beats schoolbook
                if key1 not in cutoffs:
                    cutoffs[key1] = n
                last_algo = "karatsuba"

            # detect karatsuba -> strassen cutoff
            key2 = f"karatsuba_vs_strassen_d{d}"
            if last_algo == "karatsuba" and t_stras < t_karat:
                if key2 not in cutoffs:
                    cutoffs[key2] = n
                last_algo = "strassen"

    # Save cutoffs (merge with existing file but overwrite keys we computed)
    existing = {}
    if CUTOFF_FILE.exists():
        try:
            existing = json.loads(CUTOFF_FILE.read_text())
        except Exception:
            existing = {}
    existing.update(cutoffs)
    CUTOFF_FILE.write_text(json.dumps(existing, indent=2))
    print("== Suggested Cutoffs Saved ==")
    print(existing)

    # Plot results per digit
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[tune] matplotlib not installed -- skipping plots")
        return

    for d in digits_list:
        plt.figure(figsize=(6,4))
        xs = [p[0] for p in results[d]["school"]]
        plt.plot(xs, [p[1] for p in results[d]["school"]], marker='o', label='schoolbook')
        plt.plot(xs, [p[1] for p in results[d]["karatsuba"]], marker='o', label='karatsuba')
        plt.plot(xs, [p[1] for p in results[d]["strassen"]], marker='o', label='strassen')
        plt.xlabel("Matrix size (n)")
        plt.ylabel("Time (s)")
        plt.title(f"Tuner results (digits={d})")
        plt.grid(True)
        plt.legend()
        out = OUT_DIR / f"tuner_plot_d{d}.png"
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        print(f"[tune] Plot written to {out}")

if __name__ == "__main__":
    tune()
