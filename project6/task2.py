import numpy as np
import math
import matplotlib.pyplot as plt

# ----------------------------------------------------------------
# 1) Your original code pieces (Part 2) for piecewise interpolation
# ----------------------------------------------------------------

def newton_divided_diff(x_nodes, y_nodes, fprime=None):
    """
    Returns:
      coeff:  Newton polynomial coefficients
      coeffs_spline:  (unused here) partial for second-order terms
    """
    n = len(x_nodes)
    coeffs = [y_nodes[0]]
    coeffs_spline = np.zeros(n - 2) if n >= 3 else None
    y = np.array(y_nodes, dtype=float).copy()

    for j in range(1, n):
        for i in range(n - j):
            # If you had derivative constraints, you might handle them here
            # but we skip that for standard interpolation.
            y[i] = (y[i + 1] - y[i]) / (x_nodes[i + j] - x_nodes[i])
            if j == 2:
                coeffs_spline[i] = y[i]
        coeffs.append(y[0])

    coeff = np.array(coeffs)
    return coeff, coeffs_spline

def newton_polynomial(x, x_nodes, coeff):
    """
    Evaluate the Newton polynomial with given 'coeff' at x.
    """
    n = len(coeff)
    # Horner's scheme
    if np.isscalar(x):
        p = coeff[-1]
        for i in range(n - 2, -1, -1):
            p = p*(x - x_nodes[i]) + coeff[i]
        return p
    else:
        p = np.zeros_like(x, dtype=float)
        for ix, xx in enumerate(x):
            val = coeff[-1]
            for i in range(n - 2, -1, -1):
                val = val*(xx - x_nodes[i]) + coeff[i]
            p[ix] = val
        return p

def chebyshev_nodes(n, a, b, kind):
    """
    Example of Chebyshev node generation.
    """
    k = np.arange(n + 1)
    if kind == 1:
        # Chebyshev type 1
        mesh = np.cos((2*k + 1)*math.pi / (2*(n + 1)))
    elif kind == 2:
        # Chebyshev type 2
        mesh = np.cos(math.pi*k / n)
    else:
        raise ValueError("Unsupported kind in chebyshev_nodes.")
    # map from [-1,1] to [a,b]
    nodes = 0.5*(a + b) + 0.5*(b - a)*mesh
    return nodes

def piecewise_polynomial(f, a, b, num_sub, x, degree, local_method=0):
    """
    Build a piecewise polynomial of specified 'degree' on 'num_sub' subintervals
    from [a,b], interpolating the function f at local nodes.

    local_method=0 => equally spaced local nodes
    local_method>0 => Chebyshev local nodes.
    """
    # Create subinterval boundaries
    I = np.linspace(a, b, num_sub + 1)

    # For each subinterval [I[i], I[i+1]], build a local Newton polynomial
    sub_data = []
    for i in range(num_sub):
        ai = I[i]
        bi = I[i + 1]

        # Choose local nodes for interpolation
        if degree == 1:
            x_nodes = [ai, bi]
        elif degree == 2:
            if local_method == 0:
                # midpoint
                mid = 0.5*(ai + bi)
                x_nodes = [ai, mid, bi]
            else:
                # Chebyshev approach
                x_nodes = chebyshev_nodes(2, ai, bi, local_method)
        elif degree == 3:
            if local_method == 0:
                # 3 equally spaced local nodes (including endpoints)
                x_mid1 = ai + (bi - ai)/3
                x_mid2 = ai + 2*(bi - ai)/3
                x_nodes = [ai, x_mid1, x_mid2, bi]
            else:
                x_nodes = chebyshev_nodes(3, ai, bi, local_method)
        else:
            raise ValueError("Only degrees 1..3 shown here for brevity.")

        # Evaluate f at these local nodes
        y_nodes = [f(xi) for xi in x_nodes]

        # Build Newton polynomial
        newton_coeff, _ = newton_divided_diff(x_nodes, y_nodes)
        sub_data.append({
            'interval': (ai, bi),
            'x_nodes': np.array(x_nodes),
            'coeff': newton_coeff
        })

    # Evaluate piecewise polynomial at the array x
    x = np.atleast_1d(x)
    g = np.zeros_like(x, dtype=float)
    for ix, xx in enumerate(x):
        if xx <= a:
            sub_int = 0
        elif xx >= b:
            sub_int = num_sub - 1
        else:
            sub_int = np.searchsorted(I, xx) - 1
        data = sub_data[sub_int]
        g[ix] = newton_polynomial(xx, data['x_nodes'], data['coeff'])
    return g

# ----------------------------------------------------------------
# 2) A standard "natural cubic spline" (tridiagonal) approach
# ----------------------------------------------------------------

def natural_cubic_spline_coeffs(t_data, y_data):
    """
    Returns the array M of second derivatives at each data point,
    using natural boundary conditions (M[0] = M[-1] = 0).
    """
    n = len(t_data)
    h = np.diff(t_data)
    alpha = np.zeros(n)
    for i in range(1, n-1):
        alpha[i] = (3/h[i])*(y_data[i+1] - y_data[i]) - (3/h[i-1])*(y_data[i] - y_data[i-1])

    l = np.ones(n)
    mu = np.zeros(n)
    z = np.zeros(n)

    for i in range(1, n-1):
        l[i] = 2*(t_data[i+1] - t_data[i-1]) - h[i-1]*mu[i-1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i-1]*z[i-1]) / l[i]

    M = np.zeros(n)
    for j in range(n-2, 0, -1):
        M[j] = z[j] - mu[j]*M[j+1]

    return M

def eval_cubic_spline(t_data, y_data, M, x):
    """
    Evaluate the natural cubic spline (with second derivatives M) at x.
    """
    n = len(t_data)
    if x <= t_data[0]:
        i = 0
    elif x >= t_data[-1]:
        i = n - 2
    else:
        i = np.searchsorted(t_data, x) - 1
        if i < 0:
            i = 0
        if i > n-2:
            i = n-2

    h = t_data[i+1] - t_data[i]
    A = (t_data[i+1] - x)/h
    B = (x - t_data[i])/h
    y_val = (A*y_data[i] + B*y_data[i+1] +
             ((A**3 - A)*M[i] + (B**3 - B)*M[i+1]) * (h**2) / 6)
    return y_val

def eval_cubic_spline_deriv(t_data, y_data, M, x):
    """
    Evaluate the first derivative of the natural cubic spline at x.
    """
    n = len(t_data)
    if x <= t_data[0]:
        i = 0
    elif x >= t_data[-1]:
        i = n - 2
    else:
        i = np.searchsorted(t_data, x) - 1
        if i < 0:
            i = 0
        if i > n-2:
            i = n-2

    h = t_data[i+1] - t_data[i]
    A = (t_data[i+1] - x)/h
    B = (x - t_data[i])/h
    term1 = (y_data[i+1] - y_data[i]) / h
    term2 = -((3*A**2 - 1)*M[i]*h)/6.0
    term3 = ((3*B**2 - 1)*M[i+1]*h)/6.0
    return term1 + term2 + term3

# ----------------------------------------------------------------
# 3) Complete Task 2 and add plotting section
# ----------------------------------------------------------------

if __name__ == "__main__":
    # ------------------------------------------------------------
    # 3A. Given data (t_i, y_i)
    # ------------------------------------------------------------
    t_data = np.array([0.5, 1.0, 2.0, 4.0, 5.0, 10.0, 15.0, 20.0])
    y_data = np.array([0.04, 0.05, 0.0682, 0.0801, 0.0940, 0.0981, 0.0912, 0.0857])

    # We'll evaluate from t=0.5 to t=40.0 in increments of 0.5
    t_eval = np.arange(0.5, 40.0+0.5, 0.5)

    # ------------------------------------------------------------
    # 3B. Natural Cubic Spline Approach
    # ------------------------------------------------------------
    # 1) Compute second derivatives M at each data point
    M = natural_cubic_spline_coeffs(t_data, y_data)

    # 2) Evaluate s(t), s'(t), then f(t) and D(t)
    s_spline = []
    sprime_spline = []
    f_spline = []
    D_spline = []
    for t in t_eval:
        yy = eval_cubic_spline(t_data, y_data, M, t)
        dy = eval_cubic_spline_deriv(t_data, y_data, M, t)
        s_spline.append(yy)
        sprime_spline.append(dy)
        f_spline.append(yy + t*dy)
        D_spline.append(math.exp(-t)*yy)

    s_spline = np.array(s_spline)
    sprime_spline = np.array(sprime_spline)
    f_spline = np.array(f_spline)
    D_spline = np.array(D_spline)

    print("=== Natural Cubic Spline Results ===")
    print("   t       y(t)        f(t)         D(t)")
    for i, t in enumerate(t_eval):
        print(f"{t:5.1f}  {s_spline[i]:10.6f}  {f_spline[i]:10.6f}  {D_spline[i]:10.6f}")

    # ------------------------------------------------------------
    # 3C. Piecewise Polynomial Approach (degree=3)
    #     We define a helper function that returns the discrete y_data
    #     via a simple linear interpolation.
    # ------------------------------------------------------------
    def discrete_data_func(xx):
        return np.interp(xx, t_data, y_data)

    a = t_data[0]
    b = t_data[-1]
    num_sub = len(t_data) - 1

    g_vals = piecewise_polynomial(discrete_data_func, a, b, num_sub, t_eval, degree=3)

    # Approximate derivative using finite differences
    eps = 1e-5
    gprime_vals = (piecewise_polynomial(discrete_data_func, a, b, num_sub, t_eval+eps, 3) -
                   piecewise_polynomial(discrete_data_func, a, b, num_sub, t_eval-eps, 3)) / (2*eps)

    f_piecewise = g_vals + t_eval*gprime_vals
    D_piecewise = np.exp(-t_eval)*g_vals

    print("\n=== Piecewise Polynomial (degree=3) Results ===")
    print("   t       y(t)        f(t)         D(t)")
    for i, t in enumerate(t_eval):
        print(f"{t:5.1f}  {g_vals[i]:10.6f}  {f_piecewise[i]:10.6f}  {D_piecewise[i]:10.6f}")

    # ------------------------------------------------------------
    # 3D. Plotting the Results
    # ------------------------------------------------------------
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
