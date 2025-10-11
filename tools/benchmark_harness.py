from __future__ import annotations
import time
import json
import os
import importlib.util
from typing import Callable, Tuple, Dict, Any, Sequence
from pathlib import Path
from decimal_computer.fastdecimal import FastDecimal

DEFAULT_N = 200_000


def _make_inputs(n: int) -> Tuple[Sequence[FastDecimal], Sequence[FastDecimal]]:
    a = [FastDecimal.from_str("1.23")] * n
    b = [FastDecimal.from_str("4.56")] * n
    return a, b


def _time_fn(fn: Callable[..., Any], *args, repeats: int = 3, warmup: bool = True, **kwargs) -> float:
    if warmup:
        try:
            fn(*args, **kwargs)
        except Exception:
            pass
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


# helper to import a module by relative path under repo/tools
def _load_tool_fn(module_filename: str, attr: str):
    import sys
    from importlib import import_module

    repo_root = Path(__file__).resolve().parent.parent
    tools_pkg_root = repo_root / "tools"
    # Ensure repo root is on sys.path so 'tools' can be imported as a package
    sr = str(repo_root)
    if sr not in sys.path:
        sys.path.insert(0, sr)

    # derive package module name: tools.parallel_pool_helper from filename
    mod_name = "tools." + Path(module_filename).stem
    module = import_module(mod_name)
    return getattr(module, attr)



def run_single_baseline(n: int) -> float:
    from decimal_computer.parallel_pyworkers import mul_vectors_parallel_py as mp
    a, b = _make_inputs(n)
    def runner():
        mp(a, b, chunksize=n, max_workers=1)
    avg = _time_fn(runner, repeats=3)
    print(f"[bench] single-worker avg: {avg:.3f}s")
    return avg


def run_pool_helper(n: int, max_workers: int | None = None, chunksize: int | None = None) -> float:
    pool = _load_tool_fn("parallel_pool_helper.py", "mul_vectors_pool")
    a, b = _make_inputs(n)
    if max_workers is None:
        max_workers = os.cpu_count() or 1
    if chunksize is None:
        chunksize = max(1, n // (max_workers * 2))
    def runner():
        pool(a, b, chunksize=chunksize, max_workers=max_workers)
    avg = _time_fn(runner, repeats=3)
    print(f"[bench] pool_helper workers={max_workers} chunksize={chunksize} avg: {avg:.3f}s")
    return avg


def run_pool_shared_output(n: int, max_workers: int | None = None, chunksize: int | None = None) -> float:
    pool_shared = _load_tool_fn("parallel_pool_shared_output.py", "mul_vectors_pool_shared_output")
    a, b = _make_inputs(n)
    if max_workers is None:
        max_workers = os.cpu_count() or 1
    if chunksize is None:
        chunksize = max(1, n // (max_workers * 2))
    def runner():
        pool_shared(a, b, chunksize=chunksize, max_workers=max_workers)
    avg = _time_fn(runner, repeats=3)
    print(f"[bench] pool_shared_output workers={max_workers} chunksize={chunksize} avg: {avg:.3f}s")
    return avg


def run_pyworkers(n: int, max_workers: int | None = None, chunksize: int | None = None) -> float:
    from decimal_computer.parallel_pyworkers import mul_vectors_parallel_py as pyw
    a, b = _make_inputs(n)
    if max_workers is None:
        max_workers = os.cpu_count() or 1
    if chunksize is None:
        chunksize = max(1, n // (max_workers * 2))
    def runner():
        pyw(a, b, chunksize=chunksize, max_workers=max_workers)
    avg = _time_fn(runner, repeats=3)
    print(f"[bench] pyworkers workers={max_workers} chunksize={chunksize} avg: {avg:.3f}s")
    return avg


def run_autotune(n: int, force_rebenchmark: bool = False, sample_size: int | None = None) -> float:
    from decimal_computer.parallel_autotune import mul_vectors_parallel_auto as auto
    a, b = _make_inputs(n)
    kwargs = {}
    if force_rebenchmark:
        kwargs["force_rebenchmark"] = True
    if sample_size is not None:
        kwargs["sample_size"] = sample_size
    def runner():
        auto(a, b, **kwargs)
    avg = _time_fn(runner, repeats=1)
    print(f"[bench] autotune run avg: {avg:.3f}s")
    return avg


def run_all(n: int = DEFAULT_N, save_json: str | None = ".benchmark_results.json") -> Dict[str, float]:
    cpu = os.cpu_count() or 1
    results: Dict[str, float] = {}
    print(f"[bench] running suite n={n} cpu={cpu}")

    results["single"] = run_single_baseline(n)

    try:
        results["pool_helper_default"] = run_pool_helper(n)
    except Exception as e:
        print(f"[bench] pool_helper failed: {e}")

    try:
        results["pool_shared_default"] = run_pool_shared_output(n)
    except Exception as e:
        print(f"[bench] pool_shared_output failed: {e}")

    try:
        results["pyworkers_default"] = run_pyworkers(n)
    except Exception as e:
        print(f"[bench] pyworkers failed: {e}")

    try:
        results["autotune"] = run_autotune(n)
    except Exception as e:
        print(f"[bench] autotune failed: {e}")

    if not results:
        print("[bench] no successful results to summarize")
        return results

    best_name = min(results, key=results.get)
    print("\n[bench] summary (avg seconds):")
    for k, v in results.items():
        print(f"  {k:20s} : {v:.3f}s")
    print(f"[bench] best = {best_name} ({results[best_name]:.3f}s)")

    if save_json:
        try:
            with open(save_json, "w") as fh:
                json.dump({"n": n, "cpu": cpu, "results": results}, fh, indent=2)
            print(f"[bench] written {save_json}")
        except Exception:
            pass

    return results


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=DEFAULT_N, help="size of vectors")
    p.add_argument("--run", choices=["all", "single", "pool", "shared", "pyworkers", "autotune"], default="all")
    p.add_argument("--workers", type=int, default=None, help="override worker count for pool/pyworkers")
    p.add_argument("--chunksize", type=int, default=None, help="override chunksize")
    p.add_argument("--save", type=str, default=".benchmark_results.json")
    p.add_argument("--force-rebench", action="store_true")
    p.add_argument("--sample-size", type=int, default=None)
    args = p.parse_args()

    if args.run == "all":
        run_all(n=args.n, save_json=args.save)
        return

    if args.run == "single":
        run_single_baseline(args.n)
        return

    if args.run == "pool":
        run_pool_helper(args.n, max_workers=args.workers, chunksize=args.chunksize)
        return

    if args.run == "shared":
        run_pool_shared_output(args.n, max_workers=args.workers, chunksize=args.chunksize)
        return

    if args.run == "pyworkers":
        run_pyworkers(args.n, max_workers=args.workers, chunksize=args.chunksize)
        return

    if args.run == "autotune":
        run_autotune(args.n, force_rebenchmark=args.force_rebench, sample_size=args.sample_size)
        return


if __name__ == "__main__":
    main()
