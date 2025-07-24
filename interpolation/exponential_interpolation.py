"""
Newton's Forward and Backward Interpolation

Function: f(x) = e^x (approximated data)
Data:
    x     = [1.0, 1.5, 2.0, 2.5]
    f(x)  = [2.7183, 4.4817, 7.3891, 12.1825]

Goal:
    Estimate f(2.25) using:
    (i) Newton’s Forward Difference Interpolation
    (ii) Newton’s Backward Difference Interpolation
    (iii) Compare with exact e^2.25

"""

import numpy as np
from math import factorial, exp

# Given data
x_vals = [1.0, 1.5, 2.0, 2.5]
y_vals = [2.7183, 4.4817, 7.3891, 12.1825]
n = len(x_vals)
h = x_vals[1] - x_vals[0]

# Create difference table (forward differences)
def create_difference_table(y):
    diff_table = [y.copy()]
    for i in range(1, n):
        diff = [diff_table[-1][j+1] - diff_table[-1][j] for j in range(n - i)]
        diff_table.append(diff)
    return diff_table

# Newton's Forward Interpolation
def newton_forward(x, x0, diff_table):
    u = (x - x0) / h
    result = diff_table[0][0]
    for i in range(1, len(diff_table)):
        term = diff_table[i][0]
        for j in range(i):
            term *= (u - j)
        result += term / factorial(i)
    return result

# Newton's Backward Interpolation
def newton_backward(x, xn, diff_table):
    u = (x - xn) / h
    result = diff_table[0][-1]
    for i in range(1, len(diff_table)):
        term = diff_table[i][-1]
        for j in range(i):
            term *= (u + j)
        result += term / factorial(i)
    return result

# Construct difference table
diff_table = create_difference_table(y_vals)

# Interpolation point
x_interp = 2.25
f_exact = exp(x_interp)

# Forward Interpolation (use start point)
f_forward = newton_forward(x_interp, x_vals[0], diff_table)

# Backward Interpolation (use end point)
f_backward = newton_backward(x_interp, x_vals[-1], diff_table)

# Results
print(f"Interpolated f(2.25) using Forward Difference:  {f_forward:.5f}")
print(f"Interpolated f(2.25) using Backward Difference: {f_backward:.5f}")
print(f"Exact f(2.25) = e^2.25 = {f_exact:.5f}")

# Errors
error_forward = abs(f_exact - f_forward)
error_backward = abs(f_exact - f_backward)

print(f"\nError (Forward)  = {error_forward:.5f}")
print(f"Error (Backward) = {error_backward:.5f}")
