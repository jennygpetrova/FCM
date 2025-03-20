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
    coeffs_spline = np.zeros(n - 2) if n >= 3 else None
    y = np.array(y_nodes, dtype=float).copy()

    for j in range(1, n):
        for i in range(n - j):
            if j == 1 and fprime is not None and x_nodes[i] == x_nodes[i + 1]:
                y[i] = fprime(x_nodes[i]) / math.factorial(j)
            else:
                y[i] = (y[i + 1] - y[i]) / (x_nodes[i + j] - x_nodes[i])
            if j == 2:
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

def piecewise_polynomial(f, a, b, num_sub, x, degree, local_method, hermite=False, fprime=None):
    # Create global mesh points
    I = np.linspace(a, b, num_sub + 1)

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
def cubic_spline_coefficients(x_nodes, coeff, fprime2):
    n = len(x_nodes)
    matrix = np.zeros((n, n))
    d = np.zeros(n)
    d[0] = 2 * fprime2(x_nodes[0])
    d[-1] = 2 * fprime2(x_nodes[-1])
    matrix[0, 0] = 2
    matrix[0, 1] = 0
    matrix[n-1, n-2] = 0
    matrix[n-1, n-1] = 2

    for i in range(1, n-1):
        h = x_nodes[i] - x_nodes[i - 1]
        h_next = x_nodes[i + 1] - x_nodes[i]
        mu = h / (h + h_next)
        lam = h_next / (h + h_next)
        matrix[i, i - 1] = mu
        matrix[i, i] = 2
        matrix[i, i + 1] = lam
        d[i] = 6 * coeff[i-1]

    return matrix, d

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


def cubic_spline2_polynomial(x_nodes, t):
    if len(x_nodes) != 5:
        print('!!! Cubic B-Spline ERROR: x_nodes must have length 5 !!!')
        return None
    B = np.zeros_like(t)
    for j in range(len(t)):
        if t[j] < x_nodes[0] or t[j] > x_nodes[-1]:
            B[j] = 0
        else:
            k = np.searchsorted(x_nodes, t[j])
            h = x_nodes[k] - x_nodes[k - 1]
            poly_sum = 0.0
            for i in range(1,k):
                binom = math.comb(4, i)
                sign = (-1) ** i
                term = (x_nodes[i] - t[j]) ** 3
                poly_sum += sign * binom * term
            B[j] = poly_sum / (h ** 3)
    return B



# ------------------------------------------------------------
# Testing Functions
# ------------------------------------------------------------
# def f(x):
#     return (x ** 3) + (2 * x ** 2)
#
# def fprime(x):
#     return (3 * x ** 2) + (4 * x)
#
# def fprime2(x):
#     return (6 * x) + 4
#
# x_eval = (2, 4, 6)
# x_nodes = np.array([1, 3, 5, 7, 8])
# print("x_nodes:", x_nodes)
# y_nodes = f(x_nodes)
# print("y_nodes:", y_nodes)
#
# # Testing Newton divided differences with derivative (for Hermite)
# coeff, coeff_spline = newton_divided_diff(x_nodes, y_nodes, fprime=fprime)
# print("Newton Coefficients:", coeff)
# print("Second Order Divided Differences:", coeff_spline)
#
# # Evaluate the Newton polynomial (using full nodes as x_nodes here for a simple test)
# p = newton_polynomial(x_nodes, x_nodes, coeff)
# print("Newton Polynomial Evaluation:", p)
#
# # Testing piecewise polynomial with Hermite interpolation
# g = piecewise_polynomial(f, 0, 5, 5, x_eval, 3, 0, 2, hermite=True, fprime=fprime)
# print("Piecewise Polynomial Evaluation:", g)
#
# # Testing the cubic spline coefficient matrix
# matrix, d = cubic_spline_coefficients(x_eval, coeff_spline, fprime2)
# print("Cubic Spline Coefficient Matrix:")
# print(matrix)
# print("Cubic Spline Coefficients:")
# print(d)
#
# matrix_inv = np.linalg.inv(matrix)
# s = np.dot(matrix_inv, d)
# print("s''(x): ", s)
#
# p2 = cubic_spline1_polynomial(x_nodes, y_nodes, x_eval, s)
# print("Cubic Spline Polynomial Evaluation:", p2)
#
# p3 = piecewise_polynomial(f, 0, 5, 5, x_eval, 3, 0, 2, hermite=False, fprime=fprime2)
# print("Piecewise Polynomial Evaluation:", p3)
#
# # Example usage:
# if __name__ == '__main__':
#     x_nodes = np.linspace(-5, 5, 11)
#     y_nodes = f(x_nodes)
#
#     # Here, we assume that you have computed the interior second derivatives s_int
#     # from the spline system (for natural boundary conditions). For illustration,
#     # we use a dummy array (e.g., zeros) for the interior s. In practice, you should
#     # replace this with your computed values.
#     s_interior = [0] * (len(x_nodes) - 2)  # Replace with your computed interior s values.
#
#     # Create a dense set of evaluation points.
#     x_eval = np.linspace(x_nodes[0], x_nodes[-1], 500)
#     spline_values = cubic_spline1_polynomial(x_nodes, y_nodes, x_eval, s_interior)
#
#     # Plot the results.
#     plt.figure(figsize=(10, 6))
#     plt.plot(x_eval, f(x_eval), 'k-', label='f(x)')
#     plt.plot(x_eval, spline_values, 'r--', label='Cubic Spline')
#     plt.plot(x_nodes, y_nodes, 'ko', label='Nodes')
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.title('Cubic Spline Interpolation (Natural BC)')
#     plt.legend()
#    plt.show()


# Assume that all the functions from your provided code are defined here:
# chebyshev_nodes, barycentric_weights, barycentric_interpolation,
# newton_divided_diff, newton_polynomial, piecewise_polynomial,
# cubic_spline_coefficients, cubic_spline1_polynomial, cubic_spline2_polynomial

# -----------------------------
# Test 1: Barycentric Interpolation on a Cubic Polynomial
# -----------------------------
def g_cubic(x):
    return 1 + (2 * x) + (3 * x ** 2) + (4 * x ** 3)

def g_quadratic(x):
    return 2 + (6 * x) + (12 * x ** 2)

def g_linear(x):
    return 6 + (24 * x)

# Define the interval and nodes
a, b = -4, 4
n = 3  # degree 3 -> 4 nodes exactly
nodes = chebyshev_nodes(n, a, b, kind=2)
weights = barycentric_weights(nodes)
y_nodes = g_cubic(nodes)

# Evaluate on a dense grid
x_dense = np.linspace(a, b, 100)
p_bary = barycentric_interpolation(x_dense, nodes, weights, y_nodes)
error_bary = np.max(np.abs(g_cubic(x_dense) - p_bary))
print("Test 1: Barycentric Interpolation max error (cubic):", error_bary)

plt.figure()
plt.plot(x_dense, g_cubic(x_dense), 'k-', label="g(x) exact")
plt.plot(x_dense, p_bary, 'r--', label="Barycentric interp.")
plt.scatter(nodes, y_nodes, c='blue', zorder=5, label="Nodes")
plt.title("Barycentric Interpolation on Cubic Function")
plt.legend()
plt.show()

# -----------------------------
# Test 2: Piecewise Polynomial Interpolation on a Cubic Function
# -----------------------------
# We divide the interval into 3 subintervals.
num_sub = 6
# For a cubic function, using degree=3 on each subinterval should be exact.
p_piecewise = piecewise_polynomial(g_cubic, a, b, num_sub, x_dense, degree=3, local_method=2)
error_piecewise = np.max(np.abs(g_cubic(x_dense) - p_piecewise))
print("Test 2: Piecewise Polynomial Interpolation max error (cubic):", error_piecewise)

plt.figure()
plt.plot(x_dense, g_cubic(x_dense), 'k-', label="g(x) exact")
plt.plot(x_dense, p_piecewise, 'g--', label="Piecewise poly interp.")
plt.title("Piecewise Polynomial Interpolation on Cubic Function")
plt.legend()
plt.show()

# -----------------------------
# Test 3: Piecewise Hermite Polynomial Interpolation on a Cubic Function
# -----------------------------
# For a cubic function, using degree=3 on each subinterval should be exact.
p_piecewise2 = piecewise_polynomial(g_cubic, a, b, num_sub, x_dense, degree=3, local_method=2,
                                    hermite=True, fprime=g_quadratic)
error_piecewise2 = np.max(np.abs(g_cubic(x_dense) - p_piecewise2))
print("Test 3: Piecewise Hermite Polynomial Interpolation max error (cubic):", error_piecewise2)

plt.figure()
plt.plot(x_dense, g_cubic(x_dense), 'k-', label="g(x) exact")
plt.plot(x_dense, p_piecewise2, 'g--', label="Piecewise poly interp.")
plt.title("Piecewise Hermite Polynomial Interpolation on Cubic Function")
plt.legend()
plt.show()


# -----------------------------
# Test 4: Cubic Spline Interpolation (Spline Code 1) on a Cubic Function
# -----------------------------
# For a cubic function, the natural cubic spline (with second derivative matching at endpoints)
# should exactly reproduce the function.
# Choose a set of nodes and compute the second derivative function for g_cubic.
# Choose nodes (here we use 5 nodes for a smoother spline test)
x_nodes = np.linspace(a, b, 5)
y_nodes = g_cubic(x_nodes)
_, coeffs = newton_divided_diff(x_nodes, y_nodes)
matrix, d = cubic_spline_coefficients(x_nodes, coeffs, g_linear)
s = np.linalg.solve(matrix, d)
spline1 = cubic_spline1_polynomial(x_nodes, y_nodes, x_dense, s)
error_spline1 = np.max(np.abs(g_cubic(x_dense) - spline1))
print("Test 4: Cubic Spline (Code 1) max error (cubic):", error_spline1)

plt.figure()
plt.plot(x_dense, g_cubic(x_dense), 'k-', label="g(x) exact")
plt.plot(x_dense, spline1, 'm--', label="Cubic Spline Code 1")
plt.title("Cubic Spline (Code 1) on Cubic Function")
plt.legend()
plt.show()

# -----------------------------
# Test 5: Cubic Spline Interpolation (Spline Code 2) on a Cubic Function
# -----------------------------
# Note: cubic_spline2_polynomial expects x_nodes of length 5.
# We use the same x_nodes as in Test 3.
# Compute B-spline basis interpolation values.
spline2 = cubic_spline2_polynomial(x_nodes, x_dense)
# For a cubic function, a well‐designed B-spline interpolation should have very low error.
error_spline2 = np.max(np.abs(g_cubic(x_dense) - spline2))
print("Test 4: Cubic Spline (Code 2) max error (cubic):", error_spline2)

plt.figure()
plt.plot(x_dense, g_cubic(x_dense), 'k-', label="g(x) exact")
plt.plot(x_dense, spline2, 'c--', label="Cubic Spline Code 2")
plt.title("Cubic Spline (Code 2) on Cubic Function")
plt.legend()
plt.show()


# -----------------------------
# Test 6: Comparison on a Non-Polynomial Function (e.g., sin(x))
# -----------------------------
def f_sin(x):
    return np.sin(x)


a_sin, b_sin = 0, np.pi
x_dense_sin = np.linspace(a_sin, b_sin, 200)
n_sin = 10  # number of nodes for barycentric
nodes_sin = chebyshev_nodes(n_sin, a_sin, b_sin, kind=1)
weights_sin = barycentric_weights(nodes_sin)
y_nodes_sin = f_sin(nodes_sin)
p_bary_sin = barycentric_interpolation(x_dense_sin, nodes_sin, weights_sin, y_nodes_sin)
p_piecewise_sin = piecewise_polynomial(f_sin, a_sin, b_sin, num_sub=4, x=x_dense_sin, degree=3, local_method=1)

plt.figure()
plt.plot(x_dense_sin, f_sin(x_dense_sin), 'k-', label="sin(x) exact")
plt.plot(x_dense_sin, p_bary_sin, 'r--', label="Barycentric")
plt.plot(x_dense_sin, p_piecewise_sin, 'g--', label="Piecewise Poly")
plt.title("Comparison on sin(x)")
plt.legend()
plt.show()

error_bary_sin = np.max(np.abs(f_sin(x_dense_sin) - p_bary_sin))
error_piecewise_sin = np.max(np.abs(f_sin(x_dense_sin) - p_piecewise_sin))
print("Test 5: sin(x) Barycentric max error:", error_bary_sin)
print("Test 5: sin(x) Piecewise Poly max error:", error_piecewise_sin)


# -----------------------------
# Test 6: Randomized Testing with Random Cubic Functions
# -----------------------------
def random_cubic(x, coeffs):
    # coeffs: [a0, a1, a2, a3]
    return coeffs[0] + coeffs[1] * x + coeffs[2] * x ** 2 + coeffs[3] * x ** 3


num_tests = 10
errors = []
for _ in range(num_tests):
    # Random coefficients in a reasonable range
    coeffs = np.random.uniform(-5, 5, 4)
    f_random = lambda x: random_cubic(x, coeffs)

    # Use Chebyshev nodes for exact interpolation for a cubic
    nodes_rand = chebyshev_nodes(n, a, b, kind=1)
    weights_rand = barycentric_weights(nodes_rand)
    y_rand = f_random(nodes_rand)
    p_rand = barycentric_interpolation(x_dense, nodes_rand, weights_rand, y_rand)
    err = np.max(np.abs(f_random(x_dense) - p_rand))
    errors.append(err)

print("Test 6: Random Cubic Functions - average max error (Barycentric):", np.mean(errors))
