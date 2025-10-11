# microbench/check_view.py
from decimal_computer._c_quantize import c_mul_and_quantize_from_arrays
import array, time

print("🔍 Checking zero-copy C view path...")

a = array.array("q", [12345] * 20000)
b = array.array("q", [25000] * 20000)

t0 = time.perf_counter()
res = c_mul_and_quantize_from_arrays(a, b, 8, 4, 0)
t1 = time.perf_counter()

print("✅ Zero-copy C path works!")
print("Length:", len(res))
print("Sample:", res[:5])
print("Time:", round(t1 - t0, 4), "s")
