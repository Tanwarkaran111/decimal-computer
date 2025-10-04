# Auto-generated selector: choose_algo(n,digits)
selector_table = {
    (64,1): 'decimal_naive',
    (64,2): 'strassen',
    (64,4): 'schoolbook',
    (128,1): 'decimal_naive',
    (128,2): 'strassen',
    (128,4): 'strassen',
    (256,1): 'decimal_naive',
    (256,2): 'strassen',
    (256,4): 'strassen',
    (512,1): 'strassen',
    (512,2): 'strassen',
    (512,4): 'strassen',
}

def choose_algo(n, digits):
    # Exact-match selector. If no exact match, falls back to nearest smaller n with same digits.
    key = (int(n), int(digits))
    if key in selector_table:
        return selector_table[key]
    # fallback search: same digits, largest n <= requested
    candidates = [(k,v) for k,v in selector_table.items() if k[1]==int(digits) and k[0] <= int(n)]
    if candidates:
        best_key = max(candidates, key=lambda kv: kv[0][0])[0]
        return selector_table[best_key]
    # absolute fallback
    return 'schoolbook'
