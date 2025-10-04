# Phase 3 Performance Report

Generated: 2025-10-03T15:23:03.674736

## Overview

This report summarizes benchmark results (mean runtime + stdev) and the auto-selector table.

## Per-algo summary (sample)

- **decimal_naive** fastest sample: n=2 digits=1 mean=0.000007s
- **karatsuba** fastest sample: n=2 digits=1 mean=0.000007s
- **schoolbook** fastest sample: n=2 digits=2 mean=0.000007s
- **strassen** fastest sample: n=2 digits=1 mean=0.000003s

## Selector table (best algos)

| algo     |   n |   m |   p |   digits |   trials |   mean_seconds |   stdev_seconds |
|:---------|----:|----:|----:|---------:|---------:|---------------:|----------------:|
| strassen |   2 |   2 |   2 |        1 |        3 |       3e-06    |         1e-06   |
| strassen |   2 |   2 |   2 |        2 |        3 |       4e-06    |         2e-06   |
| strassen |   2 |   2 |   2 |        4 |        3 |       3e-06    |         0       |
| strassen |   4 |   4 |   4 |        1 |        3 |       5e-06    |         0       |
| strassen |   4 |   4 |   4 |        2 |        3 |       5e-06    |         0       |
| strassen |   4 |   4 |   4 |        4 |        3 |       5e-06    |         1e-06   |
| strassen |   8 |   8 |   8 |        1 |        3 |       9e-06    |         0       |
| strassen |   8 |   8 |   8 |        2 |        3 |       1e-05    |         0       |
| strassen |   8 |   8 |   8 |        4 |        3 |       1e-05    |         1e-06   |
| strassen |  16 |  16 |  16 |        1 |        3 |       2.9e-05  |         2e-06   |
| strassen |  16 |  16 |  16 |        2 |        3 |       2.8e-05  |         0       |
| strassen |  16 |  16 |  16 |        4 |        3 |       2.9e-05  |         3e-06   |
| strassen |  32 |  32 |  32 |        1 |        3 |       0.000108 |         1.6e-05 |
| strassen |  32 |  32 |  32 |        2 |        3 |       9.7e-05  |         4e-06   |
| strassen |  32 |  32 |  32 |        4 |        3 |       0.000104 |         8e-06   |
| strassen |  64 |  64 |  64 |        1 |        3 |       0.001687 |         6.6e-05 |
| strassen |  64 |  64 |  64 |        2 |        3 |       0.001507 |         2.2e-05 |
| strassen |  64 |  64 |  64 |        4 |        3 |       0.00166  |         7.4e-05 |

## Plots

Include generated plot files from `phase3_results/` (mean_vs_size, bar charts).

