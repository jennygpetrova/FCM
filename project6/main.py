import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import CubicSpline


"-------------------------------------------------------"
"PART 1: BARYCENTRIC INTERPOLATING POLYNOMIAL"
"-------------------------------------------------------"
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


"-------------------------------------------------------"
"PART 2: PIECEWISE INTERPOLATING POLYNOMIAL"
"-------------------------------------------------------"
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


"-------------------------------------------------------"
"PART 3: SPLINE INTERPOLATING POLYNOMIAL"
"-------------------------------------------------------"
def cubic_spline_coefficients(t_nodes, y_nodes):
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

    M = np.zeros(n)  # M = s''
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
    x = float(x)
    x0 = xi + (i - 2) * h
    x1 = xi + (i - 1) * h
    x2 = xi + i * h
    x3 = xi + (i + 1) * h
    x4 = xi + (i + 2) * h

    B = 0.0
    if x < x0 or x > x4:
        return B
    if x0 <= x < x1:
        B = ((x - x0) ** 3) / (h ** 3)
        return B
    if x1 <= x < x2:
        B = (h ** 3 + 3 * h ** 2 * (x - x1) + 3 * h * (x - x1) ** 2 - 3 * (x - x1) ** 3) / (h ** 3)
        return B
    if x2 <= x < x3:
        B = (h ** 3 + 3 * h ** 2 * (x3 - x) + 3 * h * (x3 - x) ** 2 - 3 * (x3 - x) ** 3) / (h ** 3)
        return B
    if x3 <= x <= x4:
        B = ((x4 - x) ** 3) / (h ** 3)
        return B
    return B


def cubic_bspline_interpolation(x_nodes, f, fprime):
    n = len(x_nodes) - 1
    h = x_nodes[1] - x_nodes[0]  # assuming uniform spacing
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

    alpha = np.linalg.solve(matrix, B)
    return alpha


"-------------------------------------------------------"
"TASK 1 TESTING"
"-------------------------------------------------------"
if __name__ == '__main__':

    "-------------------------------------------------------"
    "FUNCTIONS"
    "-------------------------------------------------------"
    def f(x):
        return 1 / (1 + (25 * x ** 2))

    def fprime(x):
        return - (20 * x) / ((1 + (10 * x ** 2)) ** 2)

    def fprime2(x):
        return - (20 * (1 - 30 * x ** 2)) / ((1 + (10 * x ** 2)) ** 3)

    # def f(x):
    #     return np.sin(x)
    #
    # def fprime(x):
    #     return np.cos(x)
    #
    # def fprime2(x):
    #     return - np.sin(x)
    #
    # def f(x):
    #     return x ** 3
    #
    # def fprime(x):
    #     return 3 * x ** 2
    #
    # def fprime2(x):
    #     return 6 * x

    a, b = - 10, 10
    x_dense = np.linspace(a, b, 100)

    "-------------------------------------------------------"
    "BARYCENTRIC"
    "-------------------------------------------------------"
    # n0 = 5
    # n1 = 10
    # n2 = 20
    # nodes = chebyshev_nodes(n0, a, b, kind=2)
    # nodes1 = chebyshev_nodes(n1, a, b, kind=2)
    # nodes2 = chebyshev_nodes(n2, a, b, kind=2)
    # weights = barycentric_weights(nodes)
    # weights1 = barycentric_weights(nodes1)
    # weights2 = barycentric_weights(nodes2)
    # y_nodes = f(nodes)
    # y_nodes1 = f(nodes1)
    # y_nodes2 = f(nodes2)
    #
    # p_bary = barycentric_interpolation(x_dense, nodes, weights, y_nodes)
    # p_bary1 = barycentric_interpolation(x_dense, nodes1, weights1, y_nodes1)
    # p_bary2 = barycentric_interpolation(x_dense, nodes2, weights2, y_nodes2)
    # error_bary = np.max(np.abs(f(x_dense) - p_bary))
    # error_bary1 = np.max(np.abs(f(x_dense) - p_bary1))
    # error_bary2 = np.max(np.abs(f(x_dense) - p_bary2))
    # print("Test 1: Barycentric Interpolation max error (cubic):", error_bary)
    # print("Test 2: Barycentric Interpolation max error (cubic):", error_bary1)
    # print("Test 3: Barycentric Interpolation max error (cubic):", error_bary2)
    #
    # plt.figure()
    # plt.plot(x_dense, f(x_dense), 'k-', label="f(x) exact")
    # plt.plot(x_dense, p_bary, 'r--', label="Barycentric, n=5")
    # plt.plot(x_dense, p_bary1, 'b--', label="Barycentric, n=10")
    # plt.plot(x_dense, p_bary2, 'c--', label="Barycentric, n=20")
    # plt.scatter(nodes2, y_nodes2, c='c', zorder=5, label="Nodes, n=20")
    # plt.scatter(nodes1, y_nodes1, c='b', zorder=5, label="Nodes, n=10")
    # plt.scatter(nodes, y_nodes, c='r', zorder=5, label="Nodes, n=5")
    # plt.title("Barycentric Interpolation")
    # plt.legend()
    # plt.savefig("barycentric_interpolation10.png")
    # plt.show()
    #
    # #
    # "-------------------------------------------------------"
    # "PIECEWISE"
    # "-------------------------------------------------------"
    # num_sub = 20
    # p_piecewise = piecewise_polynomial(f, a, b, num_sub, x_dense, degree=3, local_method=2)
    # error_piecewise = np.max(np.abs(f(x_dense) - p_piecewise))
    # print("Test 2: Piecewise Polynomial Interpolation max error (cubic):", error_piecewise)
    #
    # plt.figure()
    # plt.plot(x_dense, f(x_dense), 'k-', label="f(x) exact")
    # plt.plot(x_dense, p_piecewise, 'g--', label="Piecewise poly interp.")
    # plt.title("Piecewise Polynomial Interpolation")
    # plt.legend()
    # plt.savefig("piecewise_10_20.png")
    # plt.show()
    #
    #
    # "-------------------------------------------------------"
    # "PIECEWISE HERMITE"
    # "-------------------------------------------------------"
    # p_piecewise2 = piecewise_polynomial(f, a, b, num_sub, x_dense, degree=3, local_method=2,
    #                                     hermite=True, fprime=fprime)
    # error_piecewise2 = np.max(np.abs(f(x_dense) - p_piecewise2))
    # print("Test 3: Piecewise Hermite Polynomial Interpolation max error (cubic):", error_piecewise2)
    #
    # plt.figure()
    # plt.plot(x_dense, f(x_dense), 'k-', label="f(x) exact")
    # plt.plot(x_dense, p_piecewise2, 'g--', label="Piecewise poly interp.")
    # plt.title("Piecewise Hermite Polynomial Interpolation")
    # plt.legend()
    # plt.savefig("piecewise_hermite_10_20.png")
    # plt.show()


    # "-------------------------------------------------------"
    # "CUBIC SPLINE"
    # "-------------------------------------------------------"
    # p_spline = []
    # for n in (10, 50, 80):
    #     x_nodes = np.linspace(a, b, n)
    #     y_nodes = f(x_nodes)
    #     M = cubic_spline_coefficients(x_nodes, y_nodes)
    #     s_vals = []
    #     for x in x_dense:
    #         s = cubic_spline_polynomial(x_nodes, y_nodes, M, x)
    #         s_vals.append(s)
    #     s_vals = np.array(s_vals)
    #     p_spline.append(s_vals)
    #     error_spline1 = np.max(np.abs(f(x_dense) - p_spline))
    #     print(f"Test 4: Cubic Spline max error for n={n}:", error_spline1)
    #
    # plt.figure()
    # plt.plot(x_dense, f(x_dense), 'k-', label="f(x) exact")
    # plt.plot(x_dense, p_spline[0], 'c--', label="Cubic Spline n=10")
    # plt.plot(x_dense, p_spline[1], 'b--', label="Cubic Spline n=50")
    # plt.plot(x_dense, p_spline[2], 'r--', label="Cubic Spline n=80")
    # plt.title("Cubic Spline Interpolation")
    # plt.legend()
    # plt.savefig("cubic_spline_10.png")
    # plt.show()


    # "-------------------------------------------------------"
    # "CUBIC B-SPLINE"
    # "-------------------------------------------------------"
    # p_spline2 = []
    # for n in (10, 50, 80):
    #     x_nodes = np.linspace(a, b, n)
    #     y_nodes = f(x_nodes)
    #     alpha = cubic_bspline_interpolation(x_nodes, f, fprime)
    #     p_vals = np.zeros_like(x_dense)
    #     h = x_nodes[1] - x_nodes[0]
    #     x0 = x_nodes[0]
    #     N = len(x_nodes) + 2
    #
    #     for i, x in enumerate(x_dense):
    #         p = 0.0
    #         for j in range(N):
    #             index = j - 1
    #             p += alpha[j] * cubic_bspline_coefficients(index, x, x0, h)
    #         p_vals[i] = p
    #
    #     p_spline2.append(p_vals)
    #     error_spline2 = np.max(np.abs(f(x_dense) - p_vals))
    #     print(f"Test 5: Cubic B-Spline max error for n={n}:", error_spline2)
    #
    #
    # plt.plot(x_dense, f(x_dense), 'k-', label="f(x) (exact)")
    # plt.plot(x_dense, p_spline2[0], 'c--', label="Cubic B-spline n=10")
    # plt.plot(x_dense, p_spline2[1], 'b--', label="Cubic B-spline n=50")
    # plt.plot(x_dense, p_spline2[2], 'r--', label="Cubic B-spline n=80")
    # plt.title("Cubic B-spline Interpolation via B-spline Basis")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.legend()
    # plt.savefig("cubic_Bspline_10.png")
    # plt.show()

    #
    # "-------------------------------------------------------"
    # "RANDOMIZED FUNCTIONS"
    # "-------------------------------------------------------"
    # def random_cubic(x, coeffs):
    #     return coeffs[0] + (coeffs[1] * x) + (coeffs[2] * x ** 2) + (coeffs[3] * x ** 3)
    #
    # def random_quadratic(x, coeffs):
    #     return coeffs[1] + (2 * coeffs[2] * x) + (3 * coeffs[3] * x ** 2)
    #
    # def random_linear(x, coeffs):
    #     return (2 * coeffs[2]) + (6 * coeffs[3] * x)
    #
    #
    # n = 20
    # num_tests = 20
    # coeffs_list = []
    # errs_piecewise = []
    # errs_hermite = []
    # errs_spline = []
    # errs_bspline = []  # Ensure this list is defined
    # p_spline_list = []
    #
    # for test in range(num_tests):
    #     # generate 4 randomly sampled coefficients from the normal distribution
    #     coeff = np.random.randn(4) * 10
    #     coeffs_list.append(coeff)
    #     f_exact = lambda x: random_cubic(x, coeff)
    #     fprime_exact = lambda x: random_quadratic(x, coeff)
    #     y_exact = random_cubic(x_dense, coeff)
    #     x_rand = chebyshev_nodes(n, a, b, 2)
    #     y_rand = f_exact(x_rand)
    #
    #     # cubic piecewise interpolation
    #     p_piecewise = piecewise_polynomial(f_exact, a, b, 20, x_dense, degree=3, local_method=2)
    #     p_hermite = piecewise_polynomial(f_exact, a, b, 20, x_dense, degree=3, local_method=2,
    #                                      hermite=True, fprime=fprime_exact)
    #
    #     # cubic spline interpolation
    #     M = cubic_spline_coefficients(x_rand, y_rand)
    #     p_spline1 = np.array([cubic_spline_polynomial(x_rand, y_rand, M, x) for x in x_dense])
    #
    #     alpha = cubic_bspline_interpolation(x_rand, f_exact, fprime_exact)
    #     p_vals = np.zeros_like(x_dense)
    #     h = x_rand[1] - x_rand[0]
    #     x0 = x_rand[0]
    #     N = len(x_rand) + 2
    #     for i, x in enumerate(x_dense):
    #         p = 0.0
    #         for j in range(N):
    #             index = j - 1
    #             p += alpha[j] * cubic_bspline_coefficients(index, x, x0, h)
    #         p_vals[i] = p
    #     p_bspline = p_vals
    #
    #     err_piecewise = np.max(np.abs(y_exact - p_piecewise))
    #     err_hermite = np.max(np.abs(y_exact - p_hermite))
    #     err_spline = np.max(np.abs(y_exact - p_spline1))
    #     err_bspline = np.max(np.abs(f_exact(x_dense) - p_bspline))  # Use f_exact here
    #
    #     errs_piecewise.append(err_piecewise)
    #     errs_hermite.append(err_hermite)
    #     errs_spline.append(err_spline)
    #     errs_bspline.append(err_bspline)  # Append the bspline error
    #     p_spline_list.append(p_spline1)
    #
    # coeffs_list = coeffs_list[::-1]
    # print("Test\tCoefficients\t\t\tPiecewise Error\tPiecewise Hermite Error\tSpline Error\tBSpline Error")
    # for i in range(num_tests):
    #     coeff_str = np.array2string(coeffs_list[i], precision=3, separator=',')
    #     print(
    #         f"{i + 1}\t{coeff_str}\t{errs_piecewise[i]:.3e}\t{errs_hermite[i]:.3e}\t\t{errs_spline[i]:.3e}\t{errs_bspline[i]:.3e}")

    "-------------------------------------------------------"
    "TASK 2 TESTING"
    "-------------------------------------------------------"
    t_nodes = np.array([0.5, 1.0, 2.0, 4.0, 5.0, 10.0, 15.0, 20.0])
    y_nodes = np.array([0.04, 0.05, 0.0682, 0.0801, 0.0940, 0.0981, 0.0912, 0.0857])
    t_eval = np.arange(0.5, 20.0 + 0.5, 0.5)

    M = cubic_spline_coefficients(t_nodes, y_nodes)
    s_spline = []
    sprime_spline = []
    f_spline = []
    D_spline = []
    for t in t_eval:
        y_t = cubic_spline_polynomial(t_nodes, y_nodes, M, t)
        dy_t = cubic_spline_prime(t_nodes, y_nodes, M, t)
        s_spline.append(y_t)
        sprime_spline.append(dy_t)
        f_spline.append(y_t + (t * dy_t))
        D_spline.append(math.exp(-t * y_t))
    s_spline = np.array(s_spline)
    sprime_spline = np.array(sprime_spline)
    f_spline = np.array(f_spline)
    D_spline = np.array(D_spline)

    print("Natural Cubic Spline Results")
    print("   t       y(t)        f(t)         D(t)")
    for i, t in enumerate(t_eval):
        print(f"{t:5.1f}  {s_spline[i]:10.6f}  {f_spline[i]:10.6f}  {D_spline[i]:10.6f}")


    def discrete_data_func(x, t_nodes, y_nodes):
        return np.interp(x, t_nodes, y_nodes)

    f = lambda x: discrete_data_func(x, t_nodes, y_nodes)


    a = t_nodes[0]
    b = t_nodes[-1]
    num_sub = len(t_nodes) - 1
    g_vals = piecewise_polynomial(f, a, b, num_sub, t_eval, degree=1, local_method=2)
    gprime = np.zeros_like(t_eval, dtype=float)
    slopes = []
    for i in range(len(t_nodes) - 1):
        slope = (y_nodes[i + 1] - y_nodes[i]) / (t_nodes[i + 1] - t_nodes[i])
        slopes.append(slope)
    for i, x in enumerate(t_eval):
        if x <= t_nodes[0]:
            sub_int = 0
        elif x >= t_nodes[-1]:
            sub_int = len(t_nodes) - 2
        else:
            sub_int = np.searchsorted(t_nodes, x) - 1
        gprime[i] = slopes[sub_int]

    f_piecewise = g_vals + (t_eval * gprime)
    D_piecewise = np.exp(-t_eval * g_vals)

    print("\nPiecewise Polynomial (degree=1)")
    print("   t       y(t)        f(t)         D(t)")
    for i, t in enumerate(t_eval):
        print(f"{t:5.1f}  {g_vals[i]:10.6f}  {f_piecewise[i]:10.6f}  {D_piecewise[i]:10.6f}")

    # plot y(t)
    plt.plot(t_eval, s_spline, 'm--', label="Natural Cubic Spline")
    plt.plot(t_eval, g_vals, 'c--', label="Piecewise Polynomial (deg=1)")
    plt.scatter(t_nodes, y_nodes, c='r', zorder=5, label="Discrete Data")
    plt.title("Interpolated y(t)")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.legend()
    plt.savefig("y.png")
    plt.show()


    # plot f(t) = y(t) + t*y'(t)
    plt.plot(t_eval, f_spline, 'm--', label="Natural Cubic Spline")
    plt.plot(t_eval, f_piecewise, 'c--', label="Piecewise Polynomial (deg=3)")
    plt.title("Computed f(t) = y(t) + ty'(t)")
    plt.xlabel("t")
    plt.ylabel("f(t)")
    plt.legend()
    plt.savefig("f.png")
    plt.show()

    # plot D(t) = exp(-ty(t))
    plt.plot(t_eval, D_spline, 'm--', label="Natural Cubic Spline")
    plt.plot(t_eval, D_piecewise, 'c--', label="Piecewise Polynomial (deg=3)")
    plt.title("Computed D(t) = exp(-ty(t))")
    plt.xlabel("t")
    plt.ylabel("D(t)")
    plt.legend()
    plt.savefig("D.png")
    plt.show()



