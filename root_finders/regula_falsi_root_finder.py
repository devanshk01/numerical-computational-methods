import math

def f(x, num_intervals=10000):
    """Computes f(x) = ∫₀ˣ e^(-t²) dt - 0.1 using trapezoidal rule"""
    a, b = 0.0, x
    h = (b - a) / num_intervals
    integral = 0.5 * (math.exp(-a ** 2) + math.exp(-b ** 2))

    for i in range(1, num_intervals):
        t = a + i * h
        integral += math.exp(-t ** 2)

    integral *= h
    return integral - 0.1

def regula_falsi(x0, x1, tol=1e-6, max_iter=20, verbose=True):
    """Applies the Regula Falsi method to find root of f(x) = ∫₀ˣ e^(-t²) dt - 0.1"""
    for i in range(1, max_iter + 1):
        f0, f1 = f(x0), f(x1)

        if f0 * f1 > 0:
            raise ValueError("Function has same signs at x0 and x1. Root not bracketed.")

        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        fx2 = f(x2)

        if verbose:
            print(f"Iter {i:2d}: x = {x2:.10f}, f(x) = {fx2:.2e}")

        if abs(fx2) < tol:
            return round(x2, 6)

        if fx2 * f1 < 0:
            x0 = x2
        else:
            x1 = x2

    raise ValueError("Regula Falsi did not converge in given iterations.")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("Solving Integral from 0 to x of e^(-t²) dt = 0.1 using Regula Falsi Method\n")

    x0, x1 = 0.0, 1.0
    tolerance = 1e-6
    max_iterations = 20

    try:
        root = regula_falsi(x0, x1, tol=tolerance, max_iter=max_iterations)
        print(f"\nRoot (Regula Falsi method): {root:.6f} (accurate to 6 decimal places)")
    except Exception as e:
        print(f"\nError: {e}")
