# micro_bench.py
import time
import statistics
from Decimal_Computer.decimal_gemm import decimal_gemm_naive

# small matrices used in your example
A = [[12,3],[4,5]]
B = [[2,1],[10,2]]

def measure(func, repeats=200):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        times.append(t1-t0)
    return statistics.mean(times), statistics.stdev(times)

def run_once_auto():
    decimal_gemm_naive(A,B, reset_counters_before=True, return_counters=False, mul_algo="auto")

def run_once_schoolbook():
    decimal_gemm_naive(A,B, reset_counters_before=True, return_counters=False, mul_algo="schoolbook")

def run_once_karatsuba():
    decimal_gemm_naive(A,B, reset_counters_before=True, return_counters=False, mul_algo="karatsuba", karatsuba_cutoff=16)

if __name__ == "__main__":
    repeats = 500
    print(f"Micro-benchmark, repeats={repeats}")
    m_auto, s_auto = measure(run_once_auto, repeats)
    m_s, s_s = measure(run_once_schoolbook, repeats)
    m_k, s_k = measure(run_once_karatsuba, repeats)
    print("AUTO: mean {:.6f}  std {:.6f}".format(m_auto, s_auto))
    print("SCHOOLBOOK: mean {:.6f}  std {:.6f}".format(m_s, s_s))
    print("KARATSUBA: mean {:.6f}  std {:.6f}".format(m_k, s_k))
