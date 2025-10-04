# tests/test_phase3_auto_integration.py
def test_auto_gemm_small():
    from phase3.benchmarks import _find_algo_functions
    funcs = _find_algo_functions()
    assert "auto" in funcs
    auto_fn = funcs["auto"]
    # tiny matrices
    A = [[1,2],[3,4]]
    B = [[5,6],[7,8]]
    C = auto_fn(A, B)
    assert C == [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]]

def test_auto_gemm_large_scalar_correctness():
    from phase3.benchmarks import _find_algo_functions
    funcs = _find_algo_functions()
    assert "auto" in funcs
    auto_fn = funcs["auto"]
    A = [[12345678901234567890]]
    B = [[98765432109876543210]]
    C = auto_fn(A, B)
    assert C[0][0] == 12345678901234567890 * 98765432109876543210
