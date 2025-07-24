import math
import cmath  # To handle complex roots safely

def f(x):
    """Function: f(x) = cos(x) - x * exp(x)"""
    return math.cos(x) - x * math.exp(x)

def muller_method(x0, x1, x2, tol=1e-6, max_iter=20, verbose=True):
    """
    Applies Müller's method to find a root of f(x).
    
    Parameters:
        x0, x1, x2 (float): Initial approximations
        tol (float): Tolerance for stopping
        max_iter (int): Maximum iterations allowed
        verbose (bool): Print intermediate steps
    
    Returns:
        float or complex: Estimated root
    """
    for i in range(1, max_iter + 1):
        f0, f1, f2 = f(x0), f(x1), f(x2)
        h0 = x1 - x0
        h1 = x2 - x1
        δ0 = (f1 - f0) / h0
        δ1 = (f2 - f1) / h1

        a = (δ1 - δ0) / (h1 + h0)
        b = a * h1 + δ1
        c = f2

        discriminant = cmath.sqrt(b**2 - 4 * a * c)
        denom = b + discriminant if abs(b + discriminant) > abs(b - discriminant) else b - discriminant

        if denom == 0:
            raise ZeroDivisionError("Denominator in Müller method became zero.")

        dx = -2 * c / denom
        x3 = x2 + dx

        if verbose:
            print(f"Iter {i:2d}: x = {x3.real:.10f}, f(x) = {f(x3.real):.2e}")

        if abs(dx) < tol:
            return x3.real if x3.imag == 0 else x3  # Return real part if it's real
        
        x0, x1, x2 = x1, x2, x3.real

    raise ValueError(f"Did not converge in {max_iter} iterations.")

# --- MAIN ---
if __name__ == "__main__":
    print("Müller Method to Find Root of f(x) = cos(x) - x * exp(x)")

    x0, x1, x2 = -1.0, 0.0, 1.0
    tolerance = 1e-6
    max_iterations = 10

    try:
        root_muller = muller_method(x0, x1, x2, tol=tolerance, max_iter=max_iterations)
        print(f"\nEstimated root (Müller): {root_muller:.8f} (accurate to 6 decimal places)")
    except Exception as e:
        print(f"\nError: {e}")
