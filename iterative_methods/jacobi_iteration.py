"""
Jacobi Iteration Method

Solves the linear system Ax = b using the Jacobi iterative method.

Problem:
    [ 4  1  0  1 ]       [ x1 ]       [  2 ]
    [ 1  4  1  0 ]   *   [ x2 ]   =   [ -2 ]
    [ 0  1  4  1 ]       [ x3 ]       [  2 ]
    [ 1  0  1  4 ]       [ x4 ]       [ -2 ]

Initial Guess: x^(0) = [0, 0, 0, 0]
Number of Iterations: 10

"""

import numpy as np

# Input Matrix A and vector b
A = np.array([
    [4, 1, 0, 1],
    [1, 4, 1, 0],
    [0, 1, 4, 1],
    [1, 0, 1, 4]
], dtype=float)

b = np.array([2, -2, 2, -2], dtype=float)

# Initial guess
x = np.zeros(len(b))
num_iterations = 10

# Jacobi Iteration
for k in range(num_iterations):
    x_new = np.copy(x)
    for i in range(len(A)):
        sum_ = sum(A[i][j] * x[j] for j in range(len(A)) if j != i)
        x_new[i] = (b[i] - sum_) / A[i][i]
    x = x_new

# Final output
print("\nFinal approximate solution after 10 Jacobi iterations:")
x_rounded = ["%.5f" % xi for xi in x]
print(f"x = {x_rounded}")
