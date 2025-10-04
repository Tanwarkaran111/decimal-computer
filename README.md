# Decimal_Computer Prototype

This prototype simulates digit-level arithmetic and measures elementary digit multiplications/additions inside a naive GEMM implementation. It supports two digit-multiply algorithms: **schoolbook** and **Karatsuba**.

## Files of interest
- `Decimal_Computer/` — package containing `decimal_digit_starter.py` and `decimal_gemm.py`
- `demo.py` — quick demo and counters (already present)
- `benchmarks.py` — benchmark generator (creates `benchmarks.csv`)
- `plot_benchmarks.py` — plot `benchmarks.csv`
- `benchmarks_compare.py` / `plot_compare.py` — compare schoolbook vs karatsuba (op counts)
- `benchmarks_compare_time.py` / `plot_compare_time.py` — compare including wall-clock time and cutoffs
- `compute_best_cutoff.py` — pick recommended karatsuba cutoff from timing runs
- `run.py` — convenience CLI to run tasks

## Quick start
1. Install requirements:
   ```bash
   python -m pip install -r requirements.txt
