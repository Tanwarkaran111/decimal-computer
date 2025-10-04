import decimal_computer
from decimal_computer import auto_runtime

print("✅ Import succeeded")
print("Module path:", decimal_computer.__file__)

# tiny smoke test with auto_runtime
try:
    from decimal_computer.auto_runtime import decimal_gemm_auto
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    result = decimal_gemm_auto(A, B, size=2, digits=2, verbose=True)
    print("decimal_gemm_auto result:", result)
except Exception as e:
    print("⚠️ Function test failed:", e)
