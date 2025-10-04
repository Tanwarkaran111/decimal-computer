# experiments/run_large_experiment.py
"""
Orchestrate long runs: baseline (NumPy) at big sizes, and optional
decimal-optimized runs via the project's auto runtime.

Usage:
  python experiments/run_large_experiment.py --sizes 2048 4096 --repeats 3
  python experiments/run_large_experiment.py --decimal --sizes 512 1024 --digits 8 --repeats 3

Behavior:
- always runs NumPy baseline for each n
- if --decimal is provided, tries to run decimal experiments for each (n,digits)
  using decimal_computer.auto_runtime (best-effort; safe if missing)
- results appended to experiments/large_experiments.csv
- run metadata saved to experiments/last_run_meta.json
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

OUT = Path("experiments/large_experiments.csv")
META = Path("experiments/last_run_meta.json")


def run_numpy_gemm(n: int, dtype=np.float32, repeats: int = 3) -> Tuple[float, float, List[float]]:
    """Run a numpy dot baseline and return mean time, gflops, and list of times."""
    A = np.random.randn(n, n).astype(dtype)
    B = np.random.randn(n, n).astype(dtype)
    # warmup
    _ = np.dot(A, B)
    times: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = np.dot(A, B)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    mean = sum(times) / len(times)
    gflops = (2.0 * n * n * n) / (mean * 1e9) if mean > 0 else float("nan")
    return mean, gflops, times


def _try_import_auto_runtime():
    """
    Try to import the project's decimal auto runtime module.
    Return module or None.
    """
    try:
        import importlib

        mod = importlib.import_module("decimal_computer.auto_runtime")
        return mod
    except Exception:
        # some projects put auto_runtime at decimal_computer.auto_runtime or decimal_computer.auto_runtime: try both
        try:
            import importlib

            mod = importlib.import_module("decimal_computer.auto_runtime")
            return mod
        except Exception:
            return None


def run_decimal_auto_once(
    n: int, digits: int, repeats: int = 3, mod: Optional[Any] = None
) -> Tuple[Optional[float], Optional[float], Optional[List[float]], Dict[str, Any]]:
    """
    Attempt to run a decimal experiment for (n,digits) using the project's auto runtime.

    Returns:
       (mean_s, std_s, times_list_or_None, info_dict)

    info_dict will contain extra debug/notes; if run fails, mean/std/times will be None.
    """
    info: Dict[str, Any] = {"n": n, "digits": digits, "attempted": False, "notes": "", "exception": None}
    if mod is None:
        mod = _try_import_auto_runtime()
        if mod is None:
            info["notes"] = "decimal auto_runtime not available (module import failed)"
            return None, None, None, info

    # Many project runtimes provide a nicely structured API; attempt common call patterns.
    # We attempt a few call names in order: run, run_once, main_run, auto_run, auto_runtime_run, single_run
    callables_to_try = ["run", "run_once", "auto_run", "auto_runtime", "auto_runtime_run", "main", "main_run"]

    # If the module is itself callable (rare), try invoking it.
    func = None
    for name in callables_to_try:
        if hasattr(mod, name) and callable(getattr(mod, name)):
            func = getattr(mod, name)
            break

    # If we didn't find a function, but the module provides a convenience function 'cli' or 'exec', try that.
    if func is None:
        for name in ("cli", "execute", "exec", "run_experiment"):
            if hasattr(mod, name) and callable(getattr(mod, name)):
                func = getattr(mod, name)
                break

    # If still none, attempt to call a top-level 'main' with args via mod.main
    if func is None and hasattr(mod, "main") and callable(getattr(mod, "main")):
        func = getattr(mod, "main")

    if func is None:
        info["notes"] = "decimal auto_runtime module has no recognized entry point"
        return None, None, None, info

    # Now attempt call using a few common signatures (best-effort).
    # We'll try: func(n=n, digits=digits, trials=repeats),
    # then func(n, digits, repeats), then func([n], digits=...), etc.
    args_variants = [
        {"n": n, "digits": digits, "trials": repeats},
        {"n": n, "digits": digits, "trials": repeats, "repeat": repeats},
        {"n": n, "digits": digits, "repeats": repeats},
        {"size": n, "digits": digits, "trials": repeats},
        {"size": n, "digits": digits, "repeats": repeats},
        # positional attempts (func(n, digits, repeats))
        (n, digits, repeats),
        (n, digits),
        (n,),
    ]

    tried_excs = []
    info["attempted"] = True
    for variant in args_variants:
        try:
            if isinstance(variant, dict):
                res = func(**variant)
            else:
                res = func(*variant)
            # Expect res to be either:
            #  - (mean_s, std_s, times_list) 
            #  - dict with keys mean, std, times, sample, etc.
            #  - None (meaning func already printed to stdout) -> we can't parse; return None w/ note
            info["notes"] = f"called {func.__name__} with args {variant!r}"
            if res is None:
                # Can't parse structured result; caller may have printed output; return None but keep note.
                info["notes"] += " -> call returned None (no structured output)"
                return None, None, None, info
            # If a tuple-like
            if isinstance(res, (tuple, list)):
                # try to map common tuple shapes
                if len(res) >= 3:
                    mean_s = float(res[0]) if res[0] is not None else None
                    std_s = float(res[1]) if res[1] is not None else None
                    times = list(res[2]) if res[2] is not None else None
                    return mean_s, std_s, times, info
                elif len(res) == 2:
                    # (mean, std) or (mean, times)
                    a, b = res
                    if isinstance(b, (list, tuple)):
                        return float(a), None, list(b), info
                    else:
                        return float(a), float(b), None, info
                else:
                    # single scalar - treat as mean
                    try:
                        mean_s = float(res[0])
                        return mean_s, None, None, info
                    except Exception:
                        return None, None, None, info
            elif isinstance(res, dict):
                # structured dict result: look for keys
                mean_s = None
                std_s = None
                times = None
                for k in ("mean", "mean_s", "mean_time", "mean_time_s"):
                    if k in res:
                        mean_s = float(res[k])
                        break
                for k in ("std", "std_s", "std_time", "std_time_s"):
                    if k in res:
                        std_s = float(res[k])
                        break
                for k in ("times", "samples", "repeat_times"):
                    if k in res:
                        times = list(res[k])
                        break
                info.update({k: res.get(k) for k in res.keys()})
                return mean_s, std_s, times, info
            else:
                # unknown return type
                info["notes"] += f" -> returned unexpected type {type(res)}"
                return None, None, None, info
        except TypeError as te:
            tried_excs.append(("TypeError", te))
            # common problem: caller expects different arg names; continue trying variants
            continue
        except Exception as exc:
            # If function raised an exception, capture and continue trying other variants
            tried_excs.append((type(exc).__name__, exc))
            continue

    # If we reach here, no variant succeeded
    info["notes"] = "all call variants failed"
    info["exceptions"] = [(t, str(e)) for (t, e) in tried_excs]
    return None, None, None, info


def append_rows_to_csv(path: Path, rows: List[Dict[str, Any]]):
    """Append list of dict rows to CSV; create file + header if missing."""
    if not rows:
        return
    first = not path.exists()
    # make sure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as fh:
        # ensure consistent column order (union of keys across rows)
        keys = list({k for r in rows for k in r.keys()})
        writer = csv.DictWriter(fh, fieldnames=keys)
        if first:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    p = argparse.ArgumentParser(description="Run large baseline and optional decimal experiments.")
    p.add_argument("--sizes", nargs="+", type=int, default=[2048, 4096], help="Matrix sizes (n)")
    p.add_argument("--repeats", type=int, default=3, help="Repeat count for timing")
    p.add_argument("--decimal", action="store_true", help="Also run decimal auto runtime experiments (if available)")
    p.add_argument("--digits", type=int, default=8, help="Digits parameter for decimal experiments (if used). Can be single value; if decimal and multiple sizes this digits applies to all runs.")
    p.add_argument("--dtype", type=str, default="float32", help="dtype for numpy baseline (float32/float64)")
    args = p.parse_args()

    dtype = np.float32 if args.dtype == "float32" else np.float64

    all_rows: List[Dict[str, Any]] = []
    meta = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.node(),
        "sizes": args.sizes,
        "repeats": args.repeats,
        "decimal": bool(args.decimal),
        "digits": args.digits,
        "timestamp": time.time(),
    }

    # 1) run numpy baselines
    for n in args.sizes:
        print(f"Running baseline n={n} ...")
        try:
            mean, gflops, times = run_numpy_gemm(n, dtype=dtype, repeats=args.repeats)
            print(f"n={n}: mean={mean:.4f}s, gflops={gflops:.2f}")
            row = {
                "timestamp": time.time(),
                "python": platform.python_version(),
                "machine": platform.node(),
                "backend": "numpy",
                "n": n,
                "digits": None,
                "mean_s": mean,
                "std_s": float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
                "gflops": gflops,
                "repeat_times": json.dumps(times),
                "notes": "numpy baseline",
            }
            all_rows.append(row)
        except Exception as exc:
            traceback.print_exc()
            print(f"Failed numpy baseline for n={n}: {exc}")
            row = {
                "timestamp": time.time(),
                "python": platform.python_version(),
                "machine": platform.node(),
                "backend": "numpy",
                "n": n,
                "digits": None,
                "mean_s": None,
                "std_s": None,
                "gflops": None,
                "repeat_times": None,
                "notes": f"numpy failed: {exc}",
            }
            all_rows.append(row)

    # 2) optional decimal auto runtime runs
    if args.decimal:
        print("Attempting decimal auto runtime experiments...")
        mod = _try_import_auto_runtime()
        if mod is None:
            print("decimal_computer.auto_runtime not importable — skipping decimal runs.")
        for n in args.sizes:
            digits = args.digits
            print(f"Running decimal experiment n={n} digits={digits} ...")
            try:
                mean_s, std_s, times_list, info = run_decimal_auto_once(n, digits, repeats=args.repeats, mod=mod)
                # record info and fallback messaging
                notes = info.get("notes", "")
                # if info contains exceptions or extra keys, include them too
                if "exceptions" in info:
                    notes = notes + " | exceptions: " + json.dumps(info["exceptions"])
                # sometimes the decimal runner returns nothing; handle gracefully
                row = {
                    "timestamp": time.time(),
                    "python": platform.python_version(),
                    "machine": platform.node(),
                    "backend": "decimal_auto",
                    "n": n,
                    "digits": digits,
                    "mean_s": mean_s,
                    "std_s": std_s,
                    "gflops": None,
                    "repeat_times": json.dumps(times_list) if times_list is not None else None,
                    "notes": notes,
                }
                all_rows.append(row)
                if mean_s is not None:
                    print(f"decimal n={n} digits={digits} -> mean {mean_s:.6f} std {std_s if std_s is not None else 'N/A'}")
                else:
                    print(f"decimal n={n} digits={digits} -> no structured result (see notes).")
            except Exception as exc:
                traceback.print_exc()
                print(f"ERROR: decimal run failed for n={n},digits={digits}: {exc}")
                row = {
                    "timestamp": time.time(),
                    "python": platform.python_version(),
                    "machine": platform.node(),
                    "backend": "decimal_auto",
                    "n": n,
                    "digits": digits,
                    "mean_s": None,
                    "std_s": None,
                    "gflops": None,
                    "repeat_times": None,
                    "notes": f"exception: {str(exc)}",
                }
                all_rows.append(row)

    # 3) append to CSV
    if all_rows:
        # ensure consistent columns order: put meta keys first, then metrics
        # choose a canonical set of columns for file
        canonical_fields = [
            "timestamp",
            "python",
            "machine",
            "backend",
            "n",
            "digits",
            "mean_s",
            "std_s",
            "gflops",
            "repeat_times",
            "notes",
        ]
        # ensure parent folder exists
        OUT.parent.mkdir(parents=True, exist_ok=True)
        first = not OUT.exists()
        with OUT.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=canonical_fields)
            if first:
                writer.writeheader()
            for r in all_rows:
                # make sure all keys exist
                outrow = {k: r.get(k, None) for k in canonical_fields}
                writer.writerow(outrow)

        print("Saved results to", OUT)
    else:
        print("No results to save.")

    # 4) save metadata
    meta_to_write = {
        **meta,
        "completed_rows": len(all_rows),
        "timestamp_completed": time.time(),
    }
    META.parent.mkdir(parents=True, exist_ok=True)
    META.write_text(json.dumps(meta_to_write, indent=2))
    print("Saved meta to", META)


if __name__ == "__main__":
    main()
