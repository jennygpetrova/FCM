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
    for j in range(1, n):
        for i in range(n - j):
            if j==1 and fprime is not None and x_nodes[i] == x_nodes[i + 1]:
                y_nodes[i] = fprime(x_nodes[i]) / math.factorial(j)
            else:
                y_nodes[i] = (y_nodes[i + 1] - y_nodes[i]) / (x_nodes[i + j] - x_nodes[i])
        coeffs.append(y_nodes[0])
    coeff = np.array(coeffs)
    return coeff

def newton_polynomial(x, x_nodes, coeff):
    n = len(coeff)
    p = np.ones_like(x) * coeff[-1]
    for i in range(n-2, -1, -1):
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
        if hermite is not None:
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

        sub_data.append({'ai': ai, 'bi': bi, 'x_nodes': np.array(x_nodes), 'coeff': np.array(coeff)})

    x = np.atleast_1d(x)
    g = np.zeros_like(x, dtype=float)
    for i, xi in enumerate(x):
        if xi <= a:
            sub_int = 0
        elif xi >= b:
            sub_int = num_sub - 1
        else:
            sub_int = np.searchsorted(I, xi) - 1
        data = sub_data[sub_int]
        g[i] = newton_polynomial(xi, data['x_nodes'], data['coeff'])
    return g

# ------------------------------------------------------------
# Part 3: Spline Code 2: Cubic B-spline Basis Interpolation
# ------------------------------------------------------------


def f(x):
    y = np.cos(x)
    return y
def fprime(x):
    y = np.sin(x)
    return y


x_nodes = np.array([0, 1, 2])
y_nodes = f(x_nodes)
coeff = newton_divided_diff(x_nodes, y_nodes, fprime=fprime)
p = newton_polynomial(x_nodes, x_nodes, coeff)
print("coeff: ", coeff)
print("p: ", p)

g = piecewise_polynomial(f, 0, 5, 5, x_nodes, 3, 0, 2, hermite=True, fprime=fprime)
print("g: ", g)




