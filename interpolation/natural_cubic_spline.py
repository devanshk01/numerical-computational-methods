"""
Natural Cubic Spline Interpolation

Given:
    x     = [0, 1, 2, 3]
    f(x)  = [1, 4, 10, 8]
    
Boundary Conditions:
    f''(0) = f''(3) = 0  (Natural Spline)

Goal:
    - Compute cubic spline
    - Estimate f(1.5)
    - Print spline equations
    - Plot spline

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

# Step 1: Spline coefficients
def cubic_spline_coeffs(x, y):
    n = len(x) - 1
    h = [x[i+1] - x[i] for i in range(n)]

    # Construct system for internal second derivatives
    A = [[0] * (n-1) for _ in range(n-1)]
    r = [0] * (n-1)

    for i in range(n-1):
        A[i][i] = 2 * (h[i] + h[i+1])
        if i > 0:
            A[i][i-1] = h[i]
        if i < n-2:
            A[i][i+1] = h[i+1]
        r[i] = 3 * ((y[i+2] - y[i+1]) / h[i+1] - (y[i+1] - y[i]) / h[i])

    # Solve Ac = r for internal c (second derivatives)
    c = [0] * (n + 1)
    if n - 1 > 0:
        c[1:n] = solve(A, r).tolist()  # natural spline: c[0] = c[n] = 0

    # Compute b and d
    b = [0] * n
    d = [0] * n
    for i in range(n):
        b[i] = (y[i+1] - y[i]) / h[i] - h[i] * (2*c[i] + c[i+1]) / 3
        d[i] = (c[i+1] - c[i]) / (3*h[i])

    a = y[:-1]
    return a, b, c[:-1], d  # return up to index n-1 (each segment)

# Step 2: Evaluate spline at any xp
def spline_eval(x, coeffs, xp):
    a, b, c, d = coeffs
    i = next((j for j in range(len(x)-1) if x[j] <= xp < x[j+1]), len(x)-2)
    t = xp - x[i]
    return a[i] + b[i]*t + c[i]*t**2 + d[i]*t**3

# Step 3: Print each segment equation
def print_spline_equations(x, coeffs):
    a, b, c, d = coeffs
    for i in range(len(a)):
        print(f"S_{i}(x) = {a[i]:.3f} + {b[i]:.3f}(x - {x[i]}) + {c[i]:.3f}(x - {x[i]})^2 + {d[i]:.3f}(x - {x[i]})^3")

# Step 4: Plot the spline
def plot_spline(x, y, coeffs):
    x_vals = np.linspace(min(x), max(x), 200)
    y_vals = [spline_eval(x, coeffs, xi) for xi in x_vals]
    plt.plot(x, y, 'o', label='Data Points')
    plt.plot(x_vals, y_vals, label='Cubic Spline')
    plt.axvline(1.5, color='gray', linestyle='--', label='x = 1.5')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Natural Cubic Spline Interpolation')
    plt.legend()
    plt.grid(True)
    plt.show()

# Step 5: Run program
x = [0, 1, 2, 3]
y = [1, 4, 10, 8]

coeffs = cubic_spline_coeffs(x, y)
estimate = spline_eval(x, coeffs, 1.5)

print(f"\nEstimated f(1.5) ≈ {estimate:.5f}\n")
print("Cubic spline equations:")
print_spline_equations(x, coeffs)
print()
plot_spline(x, y, coeffs)
