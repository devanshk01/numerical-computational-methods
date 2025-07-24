import math

def f(x):
    """Function: f(x) = cos(x) - x * exp(x)"""
    return math.cos(x) - x * math.exp(x)

def f_prime(x):
    """First derivative: f'(x)"""
    return -math.sin(x) - math.exp(x) - x * math.exp(x)

def f_double_prime(x):
    """Second derivative: f''(x)"""
    return -math.cos(x) - 2 * math.exp(x) - x * math.exp(x)

def chebyshev_method(x0, tol=1e-6, max_iter=20, verbose=True):
    """
    Applies Chebyshev's method to find the root of a function.
    
    Parameters:
        x0 (float): Initial guess
        tol (float): Tolerance for stopping condition
        max_iter (int): Maximum number of iterations
        verbose (bool): If True, prints intermediate results
    
    Returns:
        float: Estimated root
    """
    x = x0
    for i in range(1, max_iter + 1):
        fx = f(x)
        fx_p = f_prime(x)
        fx_pp = f_double_prime(x)

        if abs(fx_p) < 1e-12:
            raise ValueError(f"Derivative too small at iteration {i}. Method fails.")

        # Chebyshev update formula
        delta = fx / fx_p
        correction = 0.5 * fx * fx_pp / (fx_p ** 2)
        x_new = x - delta * (1 + correction)

        if verbose:
            print(f"Iter {i:2d}: x = {x_new:.10f}, f(x) = {f(x_new):.2e}")

        if abs(x_new - x) < tol:
            return x_new
        
        x = x_new

    raise ValueError(f"Did not converge within {max_iter} iterations.")

if __name__ == "__main__":
    print("Chebyshev Method to Find Root of f(x) = cos(x) - x * exp(x)")

    # Set parameters
    initial_guess = 1.0
    tolerance = 1e-6
    max_iterations = 10

    try:
        root = chebyshev_method(initial_guess, tol=tolerance, max_iter=max_iterations)
        print(f"\nEstimated root: {root:.8f} (accurate to 6 decimal places)")
    except ValueError as e:
        print(f"\nError: {e}")
