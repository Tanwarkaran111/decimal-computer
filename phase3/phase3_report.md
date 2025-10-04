# Phase 3 Report: Optimization & Scaling of GEMM Library  

## Overview  
Phase 3 focused on **optimizing and scaling** our GEMM (General Matrix Multiplication) implementations.  
The main goals were:  

- Benchmarking performance across multiple algorithms and matrix sizes.  
- Introducing **auto-selection** of algorithms based on runtime profiles.  
- Tuning algorithm cutoffs and blocked multiplication sizes.  
- Adding **parallel implementations** for performance.  
- Ensuring correctness through unified wrappers and regression tests.  

---

## Benchmarking  

We systematically benchmarked:  

- **Schoolbook**  
- **Blocked Schoolbook**  
- **Karatsuba**  
- **Strassen**  
- **Decimal Naive**  

### Key Results  
- **Small matrices** → Karatsuba often wins.  
- **Medium (64–128)** → Strassen provides consistent speedups.  
- **Large (256–512)** → Blocked Schoolbook and Parallel Strassen dominate.  
- **Decimal Naive** → Baseline correctness, not optimal.  

---

### Runtime vs Size (with Error Bars)  
Digits = 1  
![Digits=1](phase3_results/phase3_mean_vs_size_d1_err.png)  

Digits = 2  
![Digits=2](phase3_results/phase3_mean_vs_size_d2_err.png)  

Digits = 4  
![Digits=4](phase3_results/phase3_mean_vs_size_d4_err.png)  

### Comparative Performance at n=512  
![n=512 Comparison](phase3_results/phase3_bar_n512_err.png)  

---

### Fastest Algorithms (from Aggregated Benchmarks)  

#### Digits = 1  
| n   | 1st | 2nd | 3rd |
|-----|-----|-----|-----|
| 64  | Strassen (0.0128s) | Schoolbook (0.0154s) | Karatsuba (0.0194s) |
| 128 | Strassen (0.102s) | Schoolbook (0.123s) | Karatsuba (0.129s) |
| 256 | Strassen (0.778s) | Decimal Naive (0.840s) | Karatsuba (0.845s) |
| 512 | Strassen (6.19s) | Decimal Naive (7.13s) | Karatsuba (7.14s) |

#### Digits = 2  
| n   | 1st | 2nd | 3rd |
|-----|-----|-----|-----|
| 64  | Strassen (0.0153s) | Schoolbook (0.0147s) | Decimal Naive (0.0163s) |
| 128 | Strassen (0.102s) | Schoolbook (0.115s) | Karatsuba (0.115s) |
| 256 | Strassen (0.878s) | Schoolbook (0.990s) | Karatsuba (1.03s) |
| 512 | Strassen (5.56s) | Schoolbook (8.20s) | Karatsuba (8.11s) |

#### Digits = 4  
| n   | 1st | 2nd | 3rd |
|-----|-----|-----|-----|
| 64  | Strassen (0.0170s) | Schoolbook (0.0183s) | Karatsuba (0.019s) |
| 128 | Strassen (0.126s) | Schoolbook (0.144s) | Karatsuba (0.142s) |
| 256 | Strassen (0.932s) | Schoolbook (1.04s) | Karatsuba (1.05s) |
| 512 | Strassen (8.45s) | Blocked Schoolbook (8.50s) | Karatsuba (8.67s) |

➡️ **Observation:** Strassen dominates at almost all scales once `n ≥ 64`.  

---

## Cutoff Tuning  

- Results in **`phase3_cutoff_table.csv`**  

### Optimal Cutoffs  
| Digits | Best Cutoff (n) |
|--------|-----------------|
| 1      | Strassen at n ≥ 64 |
| 2      | Strassen at n ≥ 32 |
| 4      | Strassen at n ≥ 32 |

---

## Block Size Tuning  

- Results in **`phase3_blocksize_results.csv`**, **`phase3_blocksize_table.json`**  

### Best Block Sizes  
| n   | Best Block |
|-----|------------|
| 64  | 64 |
| 128 | 32 |
| 256 | 64 |
| 512 | 48 |

---

## Auto-Selector  

- Reads:  
  - **`phase3_selector_table.csv`** (fastest algo per case)  
  - **`phase3_cutoff_table.csv`** (Strassen thresholds)  
  - **`phase3_blocksize_table.json`** (blocked sizes)  

### Fallback Rules  
- `n < 32` → Karatsuba  
- `n < 128` → Strassen  
- Else → Blocked Schoolbook  

---

## Correctness  

- Wrappers in **`phase3/impl_wrappers.py`** unify all signatures.  
- Regression test (`tests/test_wrappers.py`) passed:  
