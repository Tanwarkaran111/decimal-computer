# test_numba_impl.py (place in D:\Invented_library)
import traceback
import decimal_computer.numba_impl as m

print("has_numba =", getattr(m, "_has_numba", False))

try:
    result = m.multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]])
    print("multiply ->", result)
except Exception as e:
    print("multiply raised:", repr(e))
    traceback.print_exc()
