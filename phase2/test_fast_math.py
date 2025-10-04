from phase2.fast_math import digit_mul, digit_add

print("123 * 45 =", digit_mul([1,2,3], [4,5]))   # expect [5,5,3,5]
print("999 + 1 =", digit_add([9,9,9], [1]))      # expect [1,0,0,0]
