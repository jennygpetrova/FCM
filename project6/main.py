import numpy as np
import matplotlib.pyplot as plt
import math

# ------------------------------------------------------------
# Part 1: Barycentric Interpolating Polynomial (pn)
# ------------------------------------------------------------
def chebyshev_nodes(n, a, b, kind):
    if kind == 1:
        k = np.arange(n + 1)
        mesh = np.cos((2 * k + 1) * np.pi / (2 * (n + 1)))
    elif kind == 2:
        k = np.arange(n + 1)
        mesh = np.cos(np.pi * k / n)
    nodes = (a + b) / 2 + (b - a) / 2 * mesh
    return nodes

def barycentric_weights(nodes):
    n = len(nodes)
    gamma = np.ones(n)
    for j in range(n):
        for i in range(n):
            if i != j:
                gamma[j] /= (nodes[j] - nodes[i])
    return gamma

def barycentric_interpolation(x, nodes, gamma, y_i):
    n = len(nodes)
    p = np.zeros_like(x)
    w = np.ones_like(x)
    for k in range(len(x)):
        for j in range(n):
            w[k] *= (x[k] - nodes[j])
        for i in range(n):
            if np.isclose(x[k], nodes[i], atol=1e-12):
                p[k] = y_i[i]
                break
            p[k] += (w[k] * y_i[i] * gamma[i]) / (x[k] - nodes[i])
    return p

# ------------------------------------------------------------
# Part 2: Piecewise Interpolating Polynomial (gd)
# ------------------------------------------------------------
def newton_divided_diff(x_nodes, y_nodes, fprime=None):
    n = len(x_nodes)
    coeffs = [y_nodes[0]]
    coeffs_spline = np.zeros(n - 1) if n >= 3 else None
    y = np.array(y_nodes, dtype=float).copy()

    for j in range(1, n):
        for i in range(n - j):
            if j == 1 and fprime is not None and x_nodes[i] == x_nodes[i + 1]:
                y[i] = fprime(x_nodes[i]) / math.factorial(j)
            else:
                y[i] = (y[i + 1] - y[i]) / (x_nodes[i + j] - x_nodes[i])
            if j == 1:
                coeffs_spline[i] = y[i]
        coeffs.append(y[0])

    coeff = np.array(coeffs)
    return coeff, coeffs_spline

def newton_polynomial(x, x_nodes, coeff):
    n = len(coeff)
    p = np.ones_like(x) * coeff[-1]
    for i in range(n - 2, -1, -1):
        p = p * (x - x_nodes[i]) + coeff[i]
    return p

def piecewise_polynomial(f, a, b, num_sub, x, degree, global_method, local_method, hermite=False, fprime=None):
    # Create global mesh points
    if global_method == 0:
        I = np.linspace(a, b, num_sub + 1)
    else:
        I = chebyshev_nodes(num_sub, a, b, global_method)

    sub_data = []
    for i in range(num_sub):
        ai = I[i]
        bi = I[i + 1]
        # Use Hermite interpolation if hermite is True
        if hermite:
            x_nodes = [ai, ai, bi, bi]
            y_nodes = [f(ai), f(ai), f(bi), f(bi)]
            # Unpack the tuple: use only the Newton coefficients.
            newton_coeff, _ = newton_divided_diff(x_nodes, y_nodes, fprime=fprime)
            coeff = newton_coeff
        else:
            if degree == 1:
                x_nodes = [ai, bi]
            elif degree == 2:
                if local_method == 0:
                    x_mid = (ai + bi) / 2
                    x_nodes = [ai, x_mid, bi]
                else:
                    x_nodes = chebyshev_nodes(2, ai, bi, local_method)
            elif degree == 3:
                if local_method == 0:
                    x_mid1 = ai + (bi - ai) / 3
                    x_mid2 = ai + 2 * (bi - ai) / 3
                    x_nodes = [ai, x_mid1, x_mid2, bi]
                else:
                    x_nodes = chebyshev_nodes(3, ai, bi, local_method)
            y_nodes = [f(xi) for xi in x_nodes]
            newton_coeff, _ = newton_divided_diff(x_nodes, y_nodes)
            coeff = newton_coeff

        sub_data.append({
            'ai': ai,
            'bi': bi,
            'x_nodes': np.array(x_nodes),
            'coeff': coeff
        })

    x = np.atleast_1d(x)
    g = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        if x[i] <= a:
            sub_int = 0
        elif x[i] >= b:
            sub_int = num_sub - 1
        else:
            sub_int = np.searchsorted(I, x[i]) - 1
        data = sub_data[sub_int]
        g[i] = newton_polynomial(x[i], data['x_nodes'], data['coeff'])
    return g

# ------------------------------------------------------------
# Part 3: Spline Code 2: Cubic B-spline Basis Interpolation
# ------------------------------------------------------------
def cubic_spline_coefficients(x_nodes, coeff):
    n = len(x_nodes)
    matrix = np.zeros((n-1, n))  # Use zeros to initialize the matrix.
    g = np.zeros(n-1)
    for i in range(n-2):
        h = x_nodes[i] - x_nodes[i - 1]
        h_next = x_nodes[i + 1] - x_nodes[i]
        mu = h / (h + h_next)
        lam = h_next / (h + h_next)
        matrix[i, i] = mu
        matrix[i, i + 1] = 2
        matrix[i, i + 2] = lam
        g[i] = 3 * ((lam * coeff[i]) + (mu * coeff[i+1]))
    return matrix, g

def cubic_spline1_polynomial(x_nodes, y_nodes, x_eval, s):
    n = len(x_nodes)
    m = len(x_eval)

    s = list(s)
    s.insert(0, 0)
    s.append(0)
    p = np.zeros(m)

    for j in range(m):
        x = x_eval[j]
        if x <= x_nodes[0]:
            i = 1
        elif x >= x_nodes[-1]:
            i = n - 1
        else:
            for i in range(1, n):
                if x <= x_nodes[i]:
                    break
        h = x_nodes[i] - x_nodes[i - 1]
        A = (x_nodes[i] - x) / h
        B = (x - x_nodes[i - 1]) / h

        p[j] = (A * y_nodes[i - 1] + B * y_nodes[i] +
                ((A ** 3 - A) * s[i - 1] + (B ** 3 - B) * s[i]) * (h ** 2) / 6)
    return p



# ------------------------------------------------------------
# Testing Functions
# ------------------------------------------------------------
def f(x):
    return x ** 3

def fprime(x):
    return 3 * x ** 2

x_eval = (2, 4, 6)
x_nodes = np.array([1, 3, 5, 7, 8])
print("x_nodes:", x_nodes)
y_nodes = f(x_nodes)
print("y_nodes:", y_nodes)

# Testing Newton divided differences with derivative (for Hermite)
coeff, coeff_spline = newton_divided_diff(x_nodes, y_nodes, fprime=fprime)
print("Newton Coefficients:", coeff)
print("Second Order Divided Differences:", coeff_spline)

# Evaluate the Newton polynomial (using full nodes as x_nodes here for a simple test)
p = newton_polynomial(x_nodes, x_nodes, coeff)
print("Newton Polynomial Evaluation:", p)

# Testing piecewise polynomial with Hermite interpolation
g = piecewise_polynomial(f, 0, 5, 5, x_nodes, 3, 0, 2, hermite=True, fprime=fprime)
print("Piecewise Polynomial Evaluation:", g)

# Testing the cubic spline coefficient matrix
matrix, g = cubic_spline_coefficients(x_nodes, coeff_spline)
print("Cubic Spline Coefficient Matrix:")
print(matrix)

matrix_inv = np.linalg.inv(matrix)
s = np.dot(matrix_inv, g)
print("s''(x): ", s)

p2 = cubic_spline1_polynomial(x_nodes, y_nodes, x_eval, s)
print("Cubic Spline Polynomial Evaluation:", p2)

# Example usage:
if __name__ == '__main__':
    x_nodes = np.linspace(-5, 5, 11)
    y_nodes = f(x_nodes)

    # Here, we assume that you have computed the interior second derivatives s_int
    # from the spline system (for natural boundary conditions). For illustration,
    # we use a dummy array (e.g., zeros) for the interior s. In practice, you should
    # replace this with your computed values.
    s_interior = [0] * (len(x_nodes) - 2)  # Replace with your computed interior s values.

    # Create a dense set of evaluation points.
    x_eval = np.linspace(x_nodes[0], x_nodes[-1], 500)
    spline_values = cubic_spline1_polynomial(x_nodes, y_nodes, x_eval, s_interior)

    # Plot the results.
    plt.figure(figsize=(10, 6))
    plt.plot(x_eval, f(x_eval), 'k-', label='f(x)')
    plt.plot(x_eval, spline_values, 'r--', label='Cubic Spline')
    plt.plot(x_nodes, y_nodes, 'ko', label='Nodes')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Cubic Spline Interpolation (Natural BC)')
    plt.legend()
    plt.show()