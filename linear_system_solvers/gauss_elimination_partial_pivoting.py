"""
Gauss Elimination with Partial Pivoting

Solves a system of linear equations Ax = b using the Gauss elimination method
with partial pivoting.

Problem:
Solve the system of equations:

    [ 2  1  1  2 ]       [ x1 ]       [  2 ]
    [ 4  0  2  1 ]   *   [ x2 ]   =   [  3 ]
    [ 3  2  2  0 ]       [ x3 ]       [ -1 ]
    [ 1  3  2  0 ]       [ x4 ]       [ -4 ]
    
"""

import numpy as np

def gauss_elimination_partial_pivoting(A, b):
    """Applies Gauss Elimination with Partial Pivoting to solve Ax = b"""
    n = len(b)
    A = A.astype(float)
    b = b.astype(float)

    # Forward Elimination with Partial Pivoting
    for i in range(n):
        # Pivot selection
        max_row = np.argmax(np.abs(A[i:n, i])) + i
        if i != max_row:
            A[[i, max_row]] = A[[max_row, i]]
            b[[i, max_row]] = b[[max_row, i]]
        
        # Elimination step
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]

    # Back Substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    return x

# Input matrix A and vector b
A = np.array([
    [2, 1, 1, 2],
    [4, 0, 2, 1],
    [3, 2, 2, 0],
    [1, 3, 2, 0]
])
b = np.array([2, 3, -1, -4])

# Solve the system
solution = gauss_elimination_partial_pivoting(A, b)

# Output the result
print("Solution of the given system of equations is:")
print("x =", np.round(solution).astype(int).tolist())

# Expected output: [1, -1, -1, 1]
