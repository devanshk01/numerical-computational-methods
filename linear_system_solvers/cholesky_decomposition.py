"""
Cholesky Decomposition

Solves a system of linear equations Ax = b using Cholesky decomposition,
and computes the inverse of matrix A.

Problem:
Solve the system:

    [  4  -1   0   0 ]       [ x1 ]       [ 1 ]
    [ -1   4  -1   0 ]   *   [ x2 ]   =   [ 0 ]
    [  0  -1   4  -1 ]       [ x3 ]       [ 0 ]
    [  0   0  -1   4 ]       [ x4 ]       [ 0 ]

"""

import numpy as np

def cholesky_decomposition(A):
    """Performs Cholesky decomposition (A = L * L.T)"""
    n = A.shape[0]
    L = np.zeros_like(A)
    
    for i in range(n):
        for j in range(i + 1):
            sum_k = sum(L[i, k] * L[j, k] for k in range(j))
            if i == j:
                L[i, j] = np.sqrt(A[i, i] - sum_k)
            else:
                L[i, j] = (A[i, j] - sum_k) / L[j, j]
    
    return L

def forward_substitution(L, b):
    """Solves Ly = b"""
    n = L.shape[0]
    y = np.zeros(n)
    
    for i in range(n):
        y[i] = (b[i] - sum(L[i, j] * y[j] for j in range(i))) / L[i, i]
    
    return y

def backward_substitution(LT, y):
    """Solves Lᵀx = y"""
    n = LT.shape[0]
    x = np.zeros(n)
    
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(LT[i, j] * x[j] for j in range(i + 1, n))) / LT[i, i]
    
    return x

def inverse_matrix(A):
    """Computes the inverse of matrix A using Cholesky decomposition"""
    n = A.shape[0]
    A_inv = np.zeros_like(A)
    L = cholesky_decomposition(A)
    
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1
        y = forward_substitution(L, e_i)
        A_inv[:, i] = backward_substitution(L.T, y)
    
    return A_inv

# Input matrix A and vector b
A = np.array([
    [ 4, -1,  0,  0],
    [-1,  4, -1,  0],
    [ 0, -1,  4, -1],
    [ 0,  0, -1,  4]
], dtype=float)

b = np.array([1, 0, 0, 0], dtype=float)

# Decomposition and solving
L = cholesky_decomposition(A)
y = forward_substitution(L, b)
x = backward_substitution(L.T, y)
A_inv = inverse_matrix(A)

# Output
print("Solution of the given system of equations is:")
print("x = [", end="")
for i in range(len(x)):
    print(f"{x[i]:.5f}", end=" " if i != len(x) - 1 else "")
print("]")

print("\nInverse of given matrix A is:")
print(np.round(A_inv, 8))
