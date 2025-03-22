import numpy as np
import matplotlib.pyplot as plt
import math

# ------------------------------------------------------------
# Part 1: Barycentric Interpolating Polynomial
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
    y = np.array(y_nodes, dtype=float).copy()

    for j in range(1, n):
        for i in range(n - j):
            if j == 1 and fprime is not None and x_nodes[i] == x_nodes[i + 1]:
                y[i] = fprime(x_nodes[i]) / math.factorial(j)
            else:
                y[i] = (y[i + 1] - y[i]) / (x_nodes[i + j] - x_nodes[i])
        coeffs.append(y[0])

    coeff = np.array(coeffs)
    return coeff


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
            coeff = newton_divided_diff(x_nodes, y_nodes, fprime=fprime)
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
            coeff = newton_divided_diff(x_nodes, y_nodes)

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
# Part 2: Spline Codes
# ------------------------------------------------------------
def cubic_spline_coeffs(t_nodes, y_nodes):
    n = len(t_nodes)
    a = np.zeros(n)
    h = np.diff(t_nodes)
    lam = np.ones(n)
    mu = np.zeros(n)
    d = np.zeros(n)

    for i in range(1, n - 1):
        a[i] = (3 / h[i]) * (y_nodes[i + 1] - y_nodes[i]) - (3 / h[i - 1]) * (y_nodes[i] - y_nodes[i - 1])

    for i in range(1, n - 1):
        lam[i] = 2 * (t_nodes[i + 1] - t_nodes[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / lam[i]
        d[i] = (a[i] - h[i - 1] * d[i - 1]) / lam[i]

    M = np.zeros(n) # M = s''
    for j in range(n - 2, 0, -1):
        M[j] = d[j] - mu[j] * M[j + 1]
    return M


def cubic_spline_polynomial(t_nodes, y_data, M, x):
    n = len(t_nodes)

    if x <= t_nodes[0]:
        i = 0
    elif x >= t_nodes[-1]:
        i = n - 2
    else:
        i = np.searchsorted(t_nodes, x) - 1

    h = t_nodes[i + 1] - t_nodes[i]
    A = (t_nodes[i + 1] - x) / h
    B = (x - t_nodes[i]) / h
    y_val = (A * y_data[i] + B * y_data[i + 1] +
             ((A ** 3 - A) * M[i] + (B ** 3 - B) * M[i + 1]) * (h ** 2) / 6)
    return y_val


def cubic_spline_prime(t_nodes, y_nodes, M, x):
    n = len(t_nodes)

    if x <= t_nodes[0]:
        i = 0
    elif x >= t_nodes[-1]:
        i = n - 2
    else:
        i = np.searchsorted(t_nodes, x) - 1

    h = t_nodes[i + 1] - t_nodes[i]
    A = (t_nodes[i + 1] - x) / h
    B = (x - t_nodes[i]) / h
    term1 = (y_nodes[i + 1] - y_nodes[i]) / h
    term2 = -((3 * A ** 2 - 1) * M[i] * h) / 6.0
    term3 = ((3 * B ** 2 - 1) * M[i + 1] * h) / 6.0
    sum = term1 + term2 + term3
    return sum


def cubic_bspline_coefficients(i, x, xi, h):
    x0 = xi + (i - 2) * h
    x1 = xi + (i - 1) * h
    x2 = xi + i * h
    x3 = xi + (i + 1) * h
    x4 = xi + (i + 2) * h

    K = 0.0
    if x < x0 or x > x4:
        return K
    if x0 <= x < x1:
        K = ((x - x0) ** 3) / (h ** 3)
        return K
    if x1 <= x < x2:
        K = (h ** 3 + 3 * h ** 2 * (x - x1) + 3 * h * (x - x1) ** 2 - 3 * (x - x1) ** 3) / (h ** 3)
        return K
    if x2 <= x < x3:
        K = (h ** 3 + 3 * h ** 2 * (x3 - x) + 3 * h * (x3 - x) ** 2 - 3 * (x3 - x) ** 3) / (h ** 3)
        return K
    if x3 <= x <= x4:
        K = ((x4 - x) ** 3) / (h ** 3)
        return K
    return K


def cubic_bspline_interpolation(x_nodes, f, fprime):
    n = len(x_nodes) - 1
    h = x_nodes[1] - x_nodes[0]  # assuming uniform spacing
    x0 = x_nodes[0]

    N = n + 3
    matrix = np.zeros((N, N))
    B = np.zeros(N)

    matrix[0, 0] = -3.0 / h
    matrix[0, 2] = 3.0 / h
    matrix[1, 0] = 1.0
    matrix[1, 1] = 4.0
    matrix[1, 2] = 1.0
    matrix[n + 1, n] = -3.0 / h
    matrix[n + 1, n + 2] = 3.0 / h
    matrix[n + 2, n] = 1.0
    matrix[n + 2, n + 1] = 4.0
    matrix[n + 2, n + 2] = 1.0

    B[0] = fprime(x_nodes[0])
    B[1] = f(x_nodes[0])
    B[n + 1] = fprime(x_nodes[-1])
    B[n + 2] = f(x_nodes[-1])

    for i in range(1, n):
        row = i + 1
        matrix[row, i] = 1.0
        matrix[row, i + 1] = 4.0
        matrix[row, i + 2] = 1.0
        B[row] = f(x_nodes[i])

    a = np.linalg.solve(matrix, B)
    s_vals = np.zeros(len(x_nodes))
    for i, x in enumerate(x_nodes):
        s_val = 0.0
        for j in range(N):
            index = j - 1
            s_val += a[j] * cubic_bspline_coefficients(index, x, x0, h)
        s_vals[i] = s_val
    return s_vals, a


def f(x):
    return x**3

a, b = -10, 10
x_dense = np.linspace(a, b, 100)
x_nodes = np.linspace(a, b, 50)
y_nodes = f(x_nodes)
M = cubic_spline_coeffs(x_nodes, y_nodes)
splines = []
for x in x_dense:
    s = cubic_spline_polynomial(x_nodes, y_nodes, M, x)
    splines.append(s)
splines = np.array(splines)

plt.figure()
plt.plot(x_dense, f(x_dense), 'k-', label="f(x) exact")
plt.plot(x_dense, splines, 'c--', label="Cubic Spline n=50")
plt.title("Cubic Spline Interpolation")
plt.legend()
plt.savefig("cubic_spline_1.png")
plt.show()

# ------------------------------------------------------------
# Main Section: Task 2 Implementation and Plotting
# ------------------------------------------------------------
if __name__ == "__main__":
    # -----------------------------
    # 1. Given data (t_i, y_i)
    # -----------------------------
    t_nodes = np.array([0.5, 1.0, 2.0, 4.0, 5.0, 10.0, 15.0, 20.0])
    y_data = np.array([0.04, 0.05, 0.0682, 0.0801, 0.0940, 0.0981, 0.0912, 0.0857])

    # Evaluation grid: t from 0.5 to 40.0 in steps of 0.5
    t_eval = np.arange(0.5, 20.0 + 0.5, 0.5)

    # -----------------------------
    # 2. Natural Cubic Spline Approach
    # -----------------------------
    # Compute second derivatives M at data points
    M = cubic_spline_coeffs(t_nodes, y_data)

    # Evaluate s(t), its derivative, f(t)=s(t)+t*s'(t) and D(t)=exp(-t)*s(t)
    s_spline = []
    sprime_spline = []
    f_spline = []
    D_spline = []
    for t in t_eval:
        yy = cubic_spline_polynomial(t_nodes, y_data, M, t)
        dy = cubic_spline_prime(t_nodes, y_data, M, t)
        s_spline.append(yy)
        sprime_spline.append(dy)
        f_spline.append(yy + t * dy)
        D_spline.append(math.exp(-t) * yy)
    s_spline = np.array(s_spline)
    sprime_spline = np.array(sprime_spline)
    f_spline = np.array(f_spline)
    D_spline = np.array(D_spline)

    # Print natural cubic spline results (for inspection)
    print("=== Natural Cubic Spline Results ===")
    print("   t       y(t)        f(t)         D(t)")
    for i, t in enumerate(t_eval):
        print(f"{t:5.1f}  {s_spline[i]:10.6f}  {f_spline[i]:10.6f}  {D_spline[i]:10.6f}")


    # -----------------------------
    # 3. Piecewise Polynomial Approach (degree=3)
    # -----------------------------
    # Define a helper function for discrete data interpolation (using linear interp)
    def discrete_data_func(xx):
        return np.interp(xx, t_nodes, y_data)


    a = t_nodes[0]
    b = t_nodes[-1]
    num_sub = len(t_nodes) - 1  # number of subintervals based on data points

    g_vals = piecewise_polynomial(discrete_data_func, a, b, num_sub, t_eval, degree=3, local_method=0)

    # Compute derivative via finite differences
    eps = 1e-5
    gprime_vals = (piecewise_polynomial(discrete_data_func, a, b, num_sub, t_eval + eps, 3, local_method=0) -
                   piecewise_polynomial(discrete_data_func, a, b, num_sub, t_eval - eps, 3, local_method=0)) / (2 * eps)
    f_piecewise = g_vals + t_eval * gprime_vals
    D_piecewise = np.exp(-t_eval) * g_vals

    # Print piecewise polynomial results
    print("\n=== Piecewise Polynomial (degree=3) Results ===")
    print("   t       y(t)        f(t)         D(t)")
    for i, t in enumerate(t_eval):
        print(f"{t:5.1f}  {g_vals[i]:10.6f}  {f_piecewise[i]:10.6f}  {D_piecewise[i]:10.6f}")

    # -----------------------------
    # 4. Plotting the Results
    # -----------------------------
    plt.figure(figsize=(10, 12))

    # Plot y(t)
    plt.subplot(3, 1, 1)
    plt.plot(t_eval, s_spline, 'b-', label="Natural Cubic Spline")
    plt.plot(t_eval, g_vals, 'r--', label="Piecewise Polynomial (deg=3)")
    plt.title("Interpolated y(t)")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.legend()
    plt.grid(True)

    # Plot f(t) = y(t) + t*y'(t)
    plt.subplot(3, 1, 2)
    plt.plot(t_eval, f_spline, 'b-', label="Natural Cubic Spline")
    plt.plot(t_eval, f_piecewise, 'r--', label="Piecewise Polynomial (deg=3)")
    plt.title("Computed f(t) = y(t) + t*y'(t)")
    plt.xlabel("t")
    plt.ylabel("f(t)")
    plt.legend()
    plt.grid(True)

    # Plot D(t) = exp(-t)*y(t)
    plt.subplot(3, 1, 3)
    plt.plot(t_eval, D_spline, 'b-', label="Natural Cubic Spline")
    plt.plot(t_eval, D_piecewise, 'r--', label="Piecewise Polynomial (deg=3)")
    plt.title("Computed D(t) = exp(-t)*y(t)")
    plt.xlabel("t")
    plt.ylabel("D(t)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
