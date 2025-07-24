"""
LU Decomposition (Unit Upper Triangular Matrix)

Solves a system of linear equations Ax = b using LU decomposition,
assuming the upper triangular matrix U has unit diagonal entries (U[i][i] = 1).

Problem:
Solve the system of equations:

    [  2   1  -4   1 ]       [ x1 ]       [  4 ]
    [ -4   3   5  -2 ]   *   [ x2 ]   =   [ -10 ]
    [  1  -1   1  -1 ]       [ x3 ]       [  2 ]
    [  1   3  -3   2 ]       [ x4 ]       [ -1 ]

"""

import numpy as np

def LU_decomposition(A):
    """Performs LU Decomposition with unit diagonal upper matrix (U[i][i] = 1)"""
    n = A.shape[0]
    L = np.eye(n)
    U = np.zeros_like(A, dtype=float)

    for i in range(n):
        for j in range(i, n):
            U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))
        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]
    
    return L, U

def forward_substitution(L, b):
    """Solves Ly = b for y using forward substitution"""
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - sum(L[i, j] * y[j] for j in range(i))
    return y

def backward_substitution(U, y):
    """Solves Ux = y for x using backward substitution"""
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i, j] * x[j] for j in range(i + 1, n))) / U[i, i]
    return x

# Input matrix A and vector b
A = np.array([
    [2, 1, -4, 1],
    [-4, 3, 5, -2],
    [1, -1, 1, -1],
    [1, 3, -3, 2]
], dtype=float)

b = np.array([4, -10, 2, -1], dtype=float)

# Perform LU Decomposition and solve
L, U = LU_decomposition(A)
y = forward_substitution(L, b)
x = backward_substitution(U, y)

# Output the result
print("Solution of the given system of equations is:")
print("x =", np.round(x).astype(int).tolist())

# Expected Output: [1, -1, -1, -1]
