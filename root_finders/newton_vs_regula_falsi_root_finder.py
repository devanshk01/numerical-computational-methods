import math

def f(x, terms=10):
    """Computes f(x) up to given number of terms"""
    result = 0.1
    for i in range(1, terms + 1):
        term = ((-1) ** i) * (x ** i) / (math.factorial(i) ** 2)
        result += term
    return result

def f_prime(x, terms=10):
    """Computes f'(x) up to given number of terms"""
    result = 0
    for i in range(1, terms + 1):
        term = ((-1) ** i) * i * (x ** (i - 1)) / (math.factorial(i) ** 2)
        result += term
    return result

def newton_method(x0, tol=1e-5, max_iter=10, verbose=True):
    """Newton-Raphson method"""
    x = x0
    for i in range(1, max_iter + 1):
        fx = f(x)
        fpx = f_prime(x)

        if abs(fpx) < 1e-12:
            raise ZeroDivisionError(f"Derivative too small at iteration {i}.")

        x_new = x - fx / fpx

        if verbose:
            print(f"Newton Iter {i}: x = {x_new:.10f}, f(x) = {f(x_new):.2e}")

        if abs(f(x_new)) < tol:
            return round(x_new, 5)
        x = x_new

    raise ValueError("Newton method did not converge.")

def regula_falsi_method(x0, x1, tol=1e-5, max_iter=10, verbose=True):
    """Regula Falsi (False Position) method"""
    for i in range(1, max_iter + 1):
        f0, f1 = f(x0), f(x1)

        if f0 * f1 > 0:
            raise ValueError("Root not bracketed. f(x0) and f(x1) must have opposite signs.")

        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        fx2 = f(x2)

        if verbose:
            print(f"Regula Falsi Iter {i}: x = {x2:.10f}, f(x) = {fx2:.2e}")

        if abs(fx2) < tol:
            return round(x2, 5)

        if fx2 * f1 < 0:
            x0 = x2
        else:
            x1 = x2

    raise ValueError("Regula Falsi method did not converge.")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("Root Finding for f(x) = 0.1 - x + x^2/(2!)^2 - x^3/(3!)^2 + ...")
    print("Approximated to 10 terms\n")

    tol = 1e-5
    max_iterations = 10

    print("Newton-Raphson Method:")
    x0_newton = 0.5
    root_newton = newton_method(x0_newton, tol=tol, max_iter=max_iterations)
    print(f"\nRoot (Newton's method): {root_newton:.5f} (accurate to 5 digits)\n")

    print("Regula Falsi Method:")
    x0_rf, x1_rf = 0.0, 1.0
    root_rf = regula_falsi_method(x0_rf, x1_rf, tol=tol, max_iter=max_iterations)
    print(f"\nRoot (Regula Falsi method): {root_rf:.5f} (accurate to 5 digits)\n")

    print("Comparison:")
    print(f"Newton's Method Root : {root_newton:.5f}")
    print(f"Regula Falsi Root    : {root_rf:.5f}")
    print(f"Absolute Difference  : {abs(root_newton - root_rf):.2e}")
