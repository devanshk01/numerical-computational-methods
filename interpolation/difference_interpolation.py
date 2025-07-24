"""
Forward and Backward Difference Interpolation

Given data:
    x     = [0.1, 0.2, 0.3, 0.4, 0.5]
    f(x)  = [1.40, 1.56, 1.76, 2.00, 2.28]

Goal:
    - Construct forward and backward difference tables
    - Interpolate f(0.25) and f(0.35) using both methods

"""

import numpy as np
from math import factorial

# Given data
x_vals = [0.1, 0.2, 0.3, 0.4, 0.5]
y_vals = [1.40, 1.56, 1.76, 2.00, 2.28]

n = len(x_vals)
h = x_vals[1] - x_vals[0]  # Assumes equal spacing

# Create forward difference table
def forward_difference_table(y):
    diff_table = [y.copy()]
    for level in range(1, n):
        next_diff = [diff_table[-1][i+1] - diff_table[-1][i] for i in range(n - level)]
        diff_table.append(next_diff)
    return diff_table

# Forward interpolation using Newton’s method
def forward_interpolation(x, x0, diff_table):
    u = (x - x0) / h
    result = diff_table[0][0]
    for i in range(1, len(diff_table)):
        term = diff_table[i][0]
        for j in range(i):
            term *= (u - j)
        result += term / factorial(i)
    return result

# Backward interpolation using Newton’s method
def backward_interpolation(x, xn, diff_table):
    u = (x - xn) / h
    result = diff_table[0][-1]
    for i in range(1, len(diff_table)):
        term = diff_table[i][-1]
        for j in range(i):
            term *= (u + j)
        result += term / factorial(i)
    return result

# Generate tables and interpolate
fwd_table = forward_difference_table(y_vals)

# Interpolation points
points = [0.25, 0.35]

# Forward Interpolation (near beginning)
print("Using Forward Difference Interpolation:")
for pt in points:
    value = forward_interpolation(pt, x_vals[0], fwd_table)
    print(f"f({pt}) ≈ {value:.5f}")

# Backward Interpolation (near end)
print("\nUsing Backward Difference Interpolation:")
for pt in points:
    value = backward_interpolation(pt, x_vals[-1], fwd_table)
    print(f"f({pt}) ≈ {value:.5f}")
