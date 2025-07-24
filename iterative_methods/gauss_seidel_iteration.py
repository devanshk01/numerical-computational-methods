"""
Gauss-Seidel Iteration Method

Solves the linear system Ax = b using the Gauss-Seidel iterative method.

Problem:
    [ 2 -1  0  0 ]       [ x1 ]       [ 1 ]
    [-1  2 -1  0 ]   *   [ x2 ]   =   [ 0 ]
    [ 0 -1  2 -1 ]       [ x3 ]       [ 0 ]
    [ 0  0 -1  2 ]       [ x4 ]       [ 1 ]

Initial Guess: x^(0) = [0, 0, 0, 0]
Number of Iterations: 10

Author: [Your Name]
"""

import numpy as np

# Input matrix A and vector b
A = np.array([
    [2, -1,  0,  0],
    [-1, 2, -1,  0],
    [0, -1,  2, -1],
    [0,  0, -1,  2]
], dtype=float)

b = np.array([1, 0, 0, 1], dtype=float)

# Initial guess
x = np.zeros(len(b))
num_iterations = 10

# Gauss-Seidel Iteration
for k in range(num_iterations):
    for i in range(len(A)):
        sum1 = sum(A[i][j] * x[j] for j in range(i))         # Use updated values
        sum2 = sum(A[i][j] * x[j] for j in range(i + 1, len(A)))  # Use old values
        x[i] = (b[i] - (sum1 + sum2)) / A[i][i]

# Final output
print("\nFinal approximate solution after 10 Gauss-Seidel iterations:")
x_rounded = ["%.5f" % xi for xi in x]
print(f"x = {x_rounded}")
