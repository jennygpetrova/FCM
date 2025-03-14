import numpy as np
import matplotlib.pyplot as plt
import math

# ------------------------------------------------------------
# Part 1: Barycentric Interpolating Polynomial (pn)
# ------------------------------------------------------------
def chebyshev_nodes(n, a, b, kind):
    if kind == 1:
        k = np.arange(n + 1)
        nodes = np.cos((2 * k + 1) * np.pi / (2 * (n + 1)))
    elif kind == 2:
        k = np.arange(n + 1)
        nodes = np.cos(np.pi * k / n)
    mesh = (a + b) / 2 + (b - a) / 2 * nodes
    return mesh

def compute_bary_weights(nodes):
    n = len(nodes)
    w = np.ones(n)
    for j in range(n):
        for i in range(n):
            if i != j:
                w[j] /= (nodes[j] - nodes[i])
    return w

def barycentric_interp(f, a, b, n, x, kind=1):
    nodes = chebyshev_nodes(n, a, b, kind)
    y_nodes = f(nodes)
    weights = compute_bary_weights(nodes)
    x = np.atleast_1d(x)
    P = np.zeros_like(x, dtype=float)
    for i, xi in enumerate(x):
        diff = xi - nodes
        if np.any(np.abs(diff) < 1e-14):
            idx = np.argmin(np.abs(diff))
            P[i] = y_nodes[idx]
        else:
            temp = weights / diff
            P[i] = np.sum(temp * y_nodes) / np.sum(temp)
    return P if P.size > 1 else P[0]


# ------------------------------------------------------------
# Part 2: Piecewise Interpolating Polynomial (gd)
# ------------------------------------------------------------
def newton_divided_diff(x_nodes, y_nodes, fprime=None):
    n = len(x_nodes)
    coeff = np.array(y_nodes)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            if np.isclose(x_nodes[i], x_nodes[i - j]):
                if fprime is not None and x_nodes[i] in fprime:
                    coeff[i] = fprime[x_nodes[i]] / math.factorial(j)
                else:
                    raise ValueError("Repeated node encountered without derivative info.")
            else:
                coeff[i] = (coeff[i] - coeff[i - 1]) / (x_nodes[i] - x_nodes[i - j])
    return coeff

def newton_poly(x, x_nodes, coeff):
    n = len(coeff)
    p = coeff[n - 1]
    for i in range(n - 2, -1, -1):
        p = p * (x - x_nodes[i]) + coeff[i]
    return p

def piecewise_interp(f, a, b, num_sub, x, degree=3, method='uniform', hermite=False, fprime=None):
    # Create uniform subinterval endpoints.
    subintervals = np.linspace(a, b, num_sub + 1)
    sub_data = []
    for i in range(num_sub):
        ai = subintervals[i]
        bi = subintervals[i + 1]
        if hermite:
            # For cubic Hermite: use [ai, ai, bi, bi].
            x_nodes = [ai, ai, bi, bi]
            y_nodes = [f(ai), f(ai), f(bi), f(bi)]
            if fprime is None:
                raise ValueError("fprime must be provided for Hermite interpolation.")
            fprime_dict = {ai: fprime(ai), bi: fprime(bi)}
            coeff = newton_divided_diff(x_nodes, y_nodes, fprime=fprime_dict)
        else:
            if degree == 1:
                x_nodes = [ai, bi]
            elif degree == 2:
                x_int = (ai + bi) / 2
                x_nodes = [ai, x_int, bi]
            elif degree == 3:
                if method == 'uniform':
                    x_int1 = ai + (bi - ai) / 3
                    x_int2 = ai + 2 * (bi - ai) / 3
                    x_nodes = [ai, x_int1, x_int2, bi]
                elif method == 'chebyshev':
                    x_nodes = chebyshev_nodes(3, ai, bi, kind=2)
                else:
                    raise ValueError("Unknown method for node placement.")
            else:
                raise ValueError("Degree must be 1, 2, or 3 for standard interpolation.")
            y_nodes = [f(xi) for xi in x_nodes]
            coeff = newton_divided_diff(x_nodes, y_nodes)
        sub_data.append({'ai': ai, 'bi': bi, 'x_nodes': np.array(x_nodes), 'coeff': np.array(coeff)})

    x = np.atleast_1d(x)
    P = np.zeros_like(x, dtype=float)
    for idx, xi in enumerate(x):
        if xi <= a:
            i_sub = 0
        elif xi >= b:
            i_sub = num_sub - 1
        else:
            i_sub = np.searchsorted(subintervals, xi) - 1
        data = sub_data[i_sub]
        P[idx] = newton_poly(xi, data['x_nodes'], data['coeff'])
    return P if P.size > 1 else P[0]


# ------------------------------------------------------------
# Part 3: Interpolatory Cubic Spline (Spline Code 1)
# ------------------------------------------------------------
def cubic_spline_coeffs(x, y, bc_type='clamped', bc_values=None):
    n = len(x) - 1
    h = np.diff(x)
    a = y.copy()
    A = np.zeros((n + 1, n + 1))
    rhs = np.zeros(n + 1)
    if bc_type == 'clamped':
        if bc_values is None or len(bc_values) != 2:
            raise ValueError("For clamped bc, provide bc_values = (fpa, fpb).")
        fpa, fpb = bc_values
        A[0, 0] = 2 * h[0]
        A[0, 1] = h[0]
        rhs[0] = 3 * ((a[1] - a[0]) / h[0] - fpa)
        A[n, n - 1] = h[n - 1]
        A[n, n] = 2 * h[n - 1]
        rhs[n] = 3 * (fpb - (a[n] - a[n - 1]) / h[n - 1])
    elif bc_type == 'second':
        if bc_values is None or len(bc_values) != 2:
            raise ValueError("For second derivative bc, provide bc_values = (M0, Mn).")
        M0, Mn = bc_values
        A[0, 0] = 1
        rhs[0] = M0
        A[n, n] = 1
        rhs[n] = Mn
    else:
        raise ValueError("bc_type must be 'clamped' or 'second'.")
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        rhs[i] = 3 * ((a[i + 1] - a[i]) / h[i] - (a[i] - a[i - 1]) / h[i - 1])
    c = np.linalg.solve(A, rhs)
    b_coeff = np.zeros(n)
    d = np.zeros(n)
    for i in range(n):
        b_coeff[i] = (a[i + 1] - a[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])
    return a[:-1], b_coeff, c[:-1], d

def cubic_spline_interp(x_data, y_data, x, bc_type='clamped', bc_values=None):
    x_data = np.array(x_data)
    a_coeff, b_coeff, c, d = cubic_spline_coeffs(x_data, y_data, bc_type, bc_values)
    x = np.atleast_1d(x)
    s = np.zeros_like(x, dtype=float)
    for idx, xv in enumerate(x):
        if xv <= x_data[0]:
            j = 0
        elif xv >= x_data[-1]:
            j = len(x_data) - 2
        else:
            j = np.searchsorted(x_data, xv) - 1
        dx = xv - x_data[j]
        s[idx] = a_coeff[j] + b_coeff[j] * dx + c[j] * dx**2 + d[j] * dx**3
    return s if s.size > 1 else s[0]


# ------------------------------------------------------------
# Part 3: Spline Code 2: Cubic B-spline Basis Interpolation
# ------------------------------------------------------------
def bspline_basis(x, knots, i, p):
    if p == 0:
        return 1.0 if (knots[i] <= x < knots[i + 1]) else 0.0
    else:
        denom1 = knots[i + p] - knots[i]
        term1 = 0.0
        if denom1 != 0:
            term1 = (x - knots[i]) / denom1 * bspline_basis(x, knots, i, p - 1)
        denom2 = knots[i + p + 1] - knots[i + 1]
        term2 = 0.0
        if denom2 != 0:
            term2 = (knots[i + p + 1] - x) / denom2 * bspline_basis(x, knots, i + 1, p - 1)
        return term1 + term2

def bspline_collocation_matrix(x_data, knots, p):
    N = len(knots) - p - 1
    M = len(x_data)
    A = np.zeros((M, N))
    for m in range(M):
        for i in range(N):
            A[m, i] = bspline_basis(x_data[m], knots, i, p)
    return A

def cubic_bspline_interp(x_data, y_data, x):
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    n = len(x_data)
    p = 3  # cubic
    # Construct clamped knot vector
    knots = np.concatenate((
        np.full(p + 1, x_data[0]),
        x_data[1:-1],
        np.full(p + 1, x_data[-1])
    ))
    A = bspline_collocation_matrix(x_data, knots, p)
    coeffs, residuals, rank, s = np.linalg.lstsq(A, y_data, rcond=None)
    x = np.atleast_1d(x)
    s_val = np.zeros_like(x, dtype=float)
    for j, xv in enumerate(x):
        val = 0.0
        for i in range(len(coeffs)):
            val += coeffs[i] * bspline_basis(xv, knots, i, p)
        s_val[j] = val
    return s_val if s_val.size > 1 else s_val[0]


# ------------------------------------------------------------
# Example Usage: Compare All Interpolants on f(x) = sin(x) on [0, π]
# ------------------------------------------------------------
if __name__ == '__main__':
    # Test function and its derivative (for Hermite and clamped spline).
    # def f(x):
    #     return np.sin(x)
    # def fprime(x):
    #     return np.cos(x)

    def f(x):
        y = x ** 3
        return y
    def fprime(x):
        y = 2 * (x ** 2)
        return y

    a, b = 0, 3
    x_plot = np.linspace(a, b, 400)

    # Part 1: Barycentric Interpolant using Chebyshev nodes (first kind)
    n_cheb = 10  # Use 11 nodes.
    y_bary = barycentric_interp(f, a, b, n_cheb, x_plot, kind=1)

    # Part 2: Piecewise Interpolating Polynomial
    num_sub = 4
    y_piecewise = piecewise_interp(f, a, b, num_sub, x_plot, degree=3, method='uniform', hermite=False)
    y_piecewise_hermite = piecewise_interp(f, a, b, num_sub, x_plot, hermite=True, fprime=fprime)

    # Part 3: Cubic Spline Interpolation (Spline Code 1)
    x_data = np.linspace(a, b, num_sub + 1)
    y_data = f(x_data)
    spline1 = cubic_spline_interp(x_data, y_data, x_plot, bc_type='clamped', bc_values=(fprime(a), fprime(b)))

    # Part 3: Cubic B-spline Interpolation (Spline Code 2)
    spline2 = cubic_bspline_interp(x_data, y_data, x_plot)

    # Plot the results:
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, f(x_plot), 'k-', label='f(x) = sin(x)')
    plt.plot(x_plot, y_bary, 'b--', label='Barycentric Poly (Chebyshev)')
    plt.plot(x_plot, y_piecewise, 'g-.', label='Piecewise Poly (Uniform nodes)')
    plt.plot(x_plot, y_piecewise_hermite, 'c:', label='Piecewise Cubic Hermite')
    plt.plot(x_plot, spline1, 'm--', label='Cubic Spline (Code 1)')
    plt.plot(x_plot, spline2, 'r:', label='Cubic B-Spline (Code 2)')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('Interpolated values')
    plt.title('Comparison of Interpolants and Splines')
    plt.show()
