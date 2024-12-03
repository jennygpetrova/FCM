import numpy as np
np.random.seed(123)


def nonlinear_roots(f, x1, x0, method, m=1, tol=1e-6, max_iter=100000000000):

    # Steffensen's iteration
    def steffensons(x1):
        fx = f(x1)
        if abs(fx) < tol:
            return x1
        denom = f(x1 + fx) - fx
        if abs(denom) < tol:
            raise ValueError("Division by zero encountered in Steffensen's method.")
        return x1 - (fx ** 2 / denom)

    iterations = 0
    for i in range(max_iter):
        # Regula Falsi and Secant and Newton's Methods
        if method in (1, 2):
            q = (f(x1) - f(x0)) / (x1 - x0)
            if abs(q) < tol:
                raise ValueError("Division by zero encountered in Secant/Regula Falsi method.")
            x2 = x1 - (f(x1) / q)

        elif method == 3:
            q = (f(x1) - f(x0)) / (x1 - x0)
            if abs(q) < tol:
                raise ValueError("Division by zero encountered in Secant/Regula Falsi method.")
            x2 = x1 - (m * f(x1) / q)

        # Steffensen's Method
        elif method == 4:
            x2 = steffensons(x1)

        # Update variables
        if method == 1:  # Regula Falsi
            if f(x1) * f(x0) < 0:
                x0 = x1
            x1 = x2
        elif method == 2:  # Secant Method
            x0, x1 = x1, x2
        elif method in (3, 4):  # Newton's or Steffensen's Method
            x1 = x2

        iterations += 1

        # Convergence check
        if abs(f(x2)) < tol:
            return x2, iterations

    raise RuntimeError("Maximum iterations reached without convergence.")

# Get Inputs for Higher Order Roots
def get_user_inputs1():
    print("\nHigher Order Roots!")
    print("\nConsider problems of the form (x-p) ** d.")
    p = int(input("Enter the root (p): "))
    d = int(input("Enter the multiplicity (d): "))
    x1 = int(input("Enter x1: "))
    x0 = int(input("Enter x0: "))
    return p, d, x1, x0



# Define a test function
p, d, x1, x0 = get_user_inputs1()
f = lambda x: (x - p) ** d

# Regula Falsi
root_rf, iterations_rf = nonlinear_roots(f, x1, x0, method=1)
print("Root (Regula Falsi):", root_rf)
print("Iterations:", iterations_rf)

# Secant Method
root_secant, iterations_secant = nonlinear_roots(f, x1, x0, method=2)
print("Root (Secant Method):", root_secant)
print("Iterations:", iterations_secant)

# Newton's Method
root_newton, iterations_newton = nonlinear_roots(f, x1, x0, method=3)
print("Root (Newton's Method):", root_newton)
print("Iterations:", iterations_newton)

# Modified Newton's Method
root_newton_mod, iterations_newton_mod = nonlinear_roots(f, x1, x0, method=3, m=d)
print("Root (Modified Newton's Method):", root_newton_mod)
print("Iterations:", iterations_newton_mod)

# Steffensen's Method
#root_steff = nonlinear_roots(f, x1, x0, method=4)
#print("Root (Steffensen's Method):", root_steff)
