import numpy as np
import matplotlib.pyplot as plt
import math

dtype = np.float64
# dtype = np.float32

# Functions f(x) for testing routines
def f1(x, d=9, dtype=dtype):
    y = (x - 2) ** d
    return np.asarray(y, dtype=dtype)


def f2(x, d, dtype=dtype):
    y = np.ones_like(x, dtype=dtype)
    for i in range(d + 1):
        y *= (x - i)
    return y


def f3(x, x_i, n):
    l_i = 1
    for i in range(n+1):
        for j in range(i+1):
            if j != n:
                l_i *= (x - x_i[j]) / (x_i[n] - x_i[j])
    return l_i


def f4(x, dtype=dtype):
    y = 1 / (1 + (25 * x ** 2))
    y = np.asarray(y, dtype=dtype)
    return y


# Barycentric 1 form weights and interpolation (unchanged)
def bary1_weights(x_i, dtype=dtype):
    n = len(x_i)
    gamma = np.ones(n, dtype=dtype)
    for i in range(n):
        for j in range(n):
            if i != j:
                gamma[i] /= (x_i[i] - x_i[j])
    return gamma


def bary1_interpolation(x, x_i, gamma, y_i, dtype=dtype):
    n = len(x_i)
    p = np.zeros_like(x, dtype=dtype)
    p_sum = np.zeros_like(x, dtype=dtype)
    w = np.ones_like(x, dtype=dtype)
    k_num = np.zeros_like(x, dtype=dtype)
    k_denom = np.zeros_like(x, dtype=dtype)
    kappa_1 = np.zeros_like(x, dtype=dtype)

    for k in range(len(x)):
        for i in range(n):
            # If x[k] is very close to any node, we directly assign y_i to p_sum[k].
            if np.isclose(x[k], x_i, rtol=1e-05, atol=1e-08).any():
                p[k] += y_i[i]
            else:
                w[k] *= (x[k] - x_i[i])
                p_sum[k] += y_i[i] * gamma[i] / (x[k] - x_i[i])
            p[k] += w[k] * p_sum[k]
            l_i = f3(x[k], x_i, i)
            k_num[k] += np.abs(p[k])
            k_denom[k] += l_i
        if np.abs(w[k]) < 1:
            kappa_1[k] = 1
        else:
            kappa_1[k] += np.abs(l_i)
    lambda_n = np.max(kappa_1)
    kappa_y = k_num / np.abs(k_denom)
    print("barycentric 1 -- p(x):", p)
    return p, kappa_y, lambda_n


# Barycentric 2 form weights and interpolation
def bary2_weights(flag, n, a=-1, b=1, dtype=dtype):
    beta = np.ones(n + 1, dtype=dtype)
    x_i = np.zeros(n + 1, dtype=dtype)
    if flag == 1:
        # Uniform nodes
        x_i = np.linspace(a, b, n + 1, dtype=dtype)
        beta[0] = 1
        for i in range(1, n + 1):
            beta[i] = beta[i - 1] * (-1) * (n - i + 1) / i
    elif flag == 2:
        # Chebyshev First Kind
        for i in range(n + 1):
            rad = ((2 * i) + 1) * np.pi / (2 * (n + 1))
            x_i[i] = math.cos(rad)
            beta[i] = math.sin(rad) * (-1) ** i
    elif flag == 3:
        # Chebyshev Second Kind
        for i in range(n + 1):
            rad = i * np.pi / n
            x_i[i] = math.cos(rad)
            beta[i] = 0.5 * (-1) ** i if i == 0 or i == n else (-1) ** i

    return beta, x_i


def bary2_interpolation(x, x_i, beta, y_i):
    n = len(x_i)
    num = np.zeros_like(x, dtype=np.float64)
    denom = np.zeros_like(x, dtype=np.float64)
    p = np.zeros_like(x, dtype=np.float64)
    for k in range(len(x)):
        for i in range(n):
            if np.isclose(x[k], x_i[i]):
                p[k] = y_i[i]
                break
            else:
                term = beta[i] / (x[k] - x_i[i])
                num[k] += term * y_i[i]
                denom[k] += term
        if denom[k] != 0:
            p[k] = num[k] / denom[k]
    print("barycentric 2 -- p(x):", p)
    return p


# Newton Divided Differences
def newton_divided_diff(x_i, y_i, dtype=dtype):
    n = len(x_i)
    y_diff = np.copy(y_i).astype(dtype)
    divided_diff = np.zeros((n, n), dtype=dtype)
    divided_diff[:, 0] = y_diff

    for j in range(1, n):  # Loop over columns
        for i in range(n - j):  # Compute each row
            divided_diff[i, j] = (divided_diff[i + 1, j - 1] - divided_diff[i, j - 1]) / (x_i[i + j] - x_i[i])

    return divided_diff[0, :]

    # n = len(x_i)
    # y_diff = np.zeros_like(x_i, dtype=dtype)
    # y_diff[0] = y_i[0]
    # for k in range(1, n):
    #     for i in range(n - k):
    #         y_i[i] = (y_i[i + 1] - y_i[i]) / (x_i[i + k] - x_i[i])
    #     y_diff[k] = y_i[0]
    # return y_diff, y_i


# Vectorized Horner's Rule for Newton interpolation
def horners_rule(x, x_i, y_diff):
    n = len(x_i)
    s = np.ones_like(x, dtype=dtype) * y_diff[-1]
    for i in range(n-2, -1, -1):
        s = s * (x - x_i[i]) + y_diff[i]
    return s


def ordering(x_i, flag):
    if flag == 1:
        x_i.sort()
    elif flag == 2:
        x_i.sort()
        x_i = x_i[::-1]
    elif flag == 3:
        x_remaining = np.copy(x_i)
        x_leja = [x_remaining[0]]
        x_remaining = np.delete(x_remaining, 0)
        while len(x_remaining) > 0:
            for x in x_remaining:
                prod = np.array([np.prod(np.abs(x - np.array(x_leja)))])
            index = np.argmax(prod)
            x_leja.append(x_remaining[index])
            x_remaining = np.delete(x_remaining, index)
        x_i = np.array(x_leja)
    return x_i


def evaluate_p(p, f, x, dtype=dtype):
    r = p - f(x)
    norm = np.linalg.norm(r, ord=np.inf)
    print("inf norm r:", norm)
    avg = np.mean(r)
    var = np.var(r)
    return norm, avg, var


def relative_error(p, f_vals):
    err = []
    for i in range(len(f_vals)):
        if np.abs(f_vals[i]) > 1e-14:
            err.append(np.abs(p[i] - f_vals[i]) / np.abs(f_vals[i]))
        else:
            err.append(np.abs(p[i] - f_vals[i]))
    return np.array(err)


'''------------------------ TESTER ------------------------'''
'''------------- FUNCTIONS f1(x) f2(x) f3(x) f4(x) -------------'''
n = 29
eps = np.finfo(float).eps
x_test = np.linspace(0 + (10 ** 3 * eps), 0.75 - (10 ** 3 * eps), 100)
functions = [
    f1,
    lambda x: f2(x, d=9),
    f3 ]
labels = [
    r"$(x-2)^9$",
    r"$\prod_{i=0}^9 (x-i)$",
    "Lagrange Basis Product"
]

for order_type in [1, 2, 3]:
    if order_type == 1:
        order = "Increasing"
    if order_type == 2:
        order = "Decreasing"
    if order_type == 3:
        order = "Leja"
    for f, label in zip(functions, labels):
        print(label)
        err_matrix_bary = []
        err_matrix_newt = []
        for flag in [1, 2, 3]:
            # Barycentric interpolation
            beta, x_i = bary2_weights(flag, n)
            if f == f1:
                y_i = f1(x_i)
            elif f == f2:
                y_i = f2(x_i, 9)
            else:
                y_i = f3(x_i, x_i, n)
            p_bary2 = bary2_interpolation(x_test, x_i, beta, y_i)
            if f == f1:
                y_true = f1(x_test)
            elif f == f2:
                y_true = f2(x_test, 9)
            else:
                y_true = f3(x_test, x_i, n)

            # Newton interpolation:
            x_i = ordering(x_i, order_type)
            y_diff = newton_divided_diff(x_i, y_i)
            p_newton = horners_rule(x_test, x_i, y_diff)

            # Compute errors
            err_bary = relative_error(p_bary2, y_true)
            err_matrix_bary.append(err_bary)
            err_newt = relative_error(p_newton, y_true)
            err_matrix_newt.append(err_newt)

            # Get condition number bounds
            gamma = bary1_weights(x_i)
            p_bary1, kappa_y, lambda_n = bary1_interpolation(x_test, x_i, gamma, y_i)

        # Plot errors for Barycentric interpolation
        plt.figure()
        plt.plot(x_test, err_matrix_bary[0], label="Uniform", color='blue', linestyle=':')
        plt.plot(x_test, err_matrix_bary[1], label="Chebyshev First Kind", color='red', linestyle=':')
        plt.plot(x_test, err_matrix_bary[2], label="Chebyshev Second Kind", color='green', linestyle=':')
        plt.axhline(y=lambda_n, color='k', linestyle='--', label='Error Bound $\Lambda_n$')
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("Relative Error")
        plt.title(f"Barycentric Interpolation Errors for {label}")
        plt.show()

        # Plot errors for Newton interpolation
        plt.figure()
        plt.plot(x_test, err_matrix_newt[0], label="Uniform", color='blue', linestyle=':')
        plt.plot(x_test, err_matrix_newt[1], label="Chebyshev First Kind", color='red', linestyle=':')
        plt.plot(x_test, err_matrix_newt[2], label="Chebyshev Second Kind", color='green', linestyle=':')
        plt.axhline(y=lambda_n, color='k', linestyle='--', label='Error Bound $\Lambda_n$')
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("Relative Error")
        plt.title(f"Newton Interpolation Errors for {label}, {order} order")
        plt.show()


'''------------------------ TESTER ------------------------'''
'''-------------------- FUNCTION f4(x) --------------------'''
n_values = [10, 20, 25, 29, 30]
mesh_labels = {1: "Uniform", 2: "Chebyshev First Kind", 3: "Chebyshev Second Kind"}
order_types = {1: "Increasing", 2: "Decreasing", 3: "Leja"}
colors = ["blue", "red", "green", "purple", "orange"]

# Loop over mesh types (Uniform, Chebyshev 1st Kind, Chebyshev 2nd Kind)
for flag in [1, 2, 3]:
    # **Barycentric Interpolation Error Plot**
    plt.figure(figsize=(10, 6))  # Create a figure for Barycentric errors

    for idx, n in enumerate(n_values):
        beta, x_i = bary2_weights(flag, n)
        y_i = f4(x_i)
        p_bary2 = bary2_interpolation(x_test, x_i, beta, y_i)

        y_true = f4(x_test)
        err_bary = relative_error(p_bary2, y_true)

        # Plot Barycentric Errors
        plt.plot(x_test, err_bary, label=f"Barycentric (n={n})", linestyle=':', color=colors[idx])

    plt.xlabel("x")
    plt.ylabel("Relative Error")
    plt.yscale("log")  # Log scale for better visualization
    plt.title(f"Barycentric Interpolation Errors for {mesh_labels[flag]} Mesh")
    plt.legend()
    plt.grid()
    plt.show()

    # **Newton Interpolation Error Plot**
    plt.figure(figsize=(10, 6))  # Create a separate figure for Newton errors

    for order in [1, 2, 3]:
        for idx, n in enumerate(n_values):
            beta, x_i = bary2_weights(flag, n)
            y_i = f4(x_i)
            x_i = ordering(x_i, order)
            y_diff = newton_divided_diff(x_i, y_i)
            p_newton = horners_rule(x_test, x_i, y_diff)

            y_true = f4(x_test)
            err_newt = relative_error(p_newton, y_true)

            # Plot Newton Errors
            plt.plot(x_test, err_newt, label=f"Newton (n={n})", linestyle='--', color=colors[idx])

        plt.xlabel("x")
        plt.ylabel("Relative Error")
        plt.yscale("log")  # Log scale for better visualization
        plt.title(f"Newton Interpolation Errors for {mesh_labels[flag]}, order {order_types[order]}")
        plt.legend()
        plt.grid()
        plt.show()









