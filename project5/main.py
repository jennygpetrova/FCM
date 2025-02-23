import numpy as np
import matplotlib.pyplot as plt
import math

dtype = np.float64
# dtype = np.float32

# Functions f(x) for testing routines
def f1(x, d=9):
    y = (x - 2) ** d
    return y


def f2(x, d):
    y = 1
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


def f4(x):
    y = 1 / (1 + (25 * x ** 2))
    return y


# Barycentric 1 form weights and interpolation (unchanged)
def bary1_weights(x_i, dtype=dtype):
    n = len(x_i)
    gamma = np.ones(n, dtype=dtype)
    for i in range(n):
        prod = 1
        for j in range(n):
            if i != j:
                prod *= (x_i[i] - x_i[j])
        gamma[i] = 1 / prod
    return gamma


def bary1_interpolation(x, x_i, gamma, y_i, dtype=dtype):
    n = len(x_i)
    p = np.zeros_like(x, dtype=dtype)
    w = np.ones_like(x, dtype=dtype)
    kappa_1 = np.ones_like(x, dtype=dtype)
    kappa_y = np.ones_like(x, dtype=dtype)

    for k in range(len(x)):
        sum_ab_1 = 0
        sum_ab_y = 0
        for j in range(n):
            w[k] *= (x[k] - x_i[j])
        for i in range(n):
            if np.isclose(x[k], x_i[i], atol=1e-12):
                p[k] = y_i[i]
                break
            p[k] += (w[k] * y_i[i] * gamma[i]) / (x[k] - x_i[i])
            sum_ab_1 += np.abs(w[k] * gamma[i] / (x[k] - x_i[i]))
            sum_ab_y += np.abs(w[k] * y_i[i] * gamma[i]) / (x[k] - x_i[i])
        if p[k] == 0:
            kappa_y[k] = 1
        else:
            kappa_y[k] = sum_ab_y / np.abs(p[k])
        kappa_1[k] = np.abs(sum_ab_1)

    lambda_n = np.max(kappa_1)
    h_n = np.max(kappa_y)

    return p, lambda_n, h_n


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
    num = np.zeros_like(x, dtype=dtype)
    denom = np.zeros_like(x, dtype=dtype)
    p = np.zeros_like(x, dtype=dtype)
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
    return p


# Newton Divided Differences
def newton_divided_diff(x_i, y_i, dtype=dtype):
    n = len(x_i)
    y_diff = np.copy(y_i).astype(dtype)
    coeffs = [y_diff[0]]
    for j in range(1, n):
        for i in range(n - j):
            y_diff[i] = (y_diff[i + 1] - y_diff[i]) / (x_i[i + j] - x_i[i])
        coeffs.append(y_diff[0])

    divided_diff = np.array(coeffs)
    return divided_diff


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
            prod = np.array([np.prod(np.abs(x - np.array(x_leja))) for x in x_remaining])
            index = np.argmax(prod)
            x_leja.append(x_remaining[index])
            x_remaining = np.delete(x_remaining, index)
        x_i = np.array(x_leja)
    return x_i


def evaluate_p(p, y):
    r = p - y
    norm = np.linalg.norm(r, ord=np.inf)
    print("inf norm r:", norm)
    avg = np.mean(r)
    var = np.var(r)
    return norm, avg, var


def relative_error(p, y_true):
    rel_error = np.zeros_like(p, dtype=dtype)
    for i in range(len(y_true)):
        if np.isclose(y_true[i], 0, atol=1e-12):
            rel_error[i] = np.abs(p[i] - y_true[i])
        else:
            rel_error[i] = np.abs(p[i] - y_true[i]) / np.abs(y_true[i])
    return rel_error


'''------------------------ TESTER ------------------------'''
'''------------- FUNCTIONS f1(x) f2(x) f3(x) -------------'''
n = 29
eps = np.finfo(float).eps
x_test = np.linspace(-1 + (10 ** 3 * eps), 1 - (10 ** 3 * eps), 100)
functions = [f1,f2,f3]
labels = [r"$(x-2)^9$", r"$\prod_{i=0}^9 (x-i)$", "Lagrange Basis Product"]
condition_table_bary2 = []
condition_table_newt = []

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
                y_true = f1(x_test)
            elif f == f2:
                y_i = f2(x_i, 9)
                y_true = f2(x_test, 9)
            else:
                y_i = f3(x_i, x_i, n)
                y_true = f3(x_test, x_i, n)

            p_bary2 = bary2_interpolation(x_test, x_i, beta, y_i)

            gamma = bary1_weights(x_i)
            _, lambda_val, h_val = bary1_interpolation(x_test, x_i, gamma, y_i)
            condition_table_bary2.append({
                "Function": label,
                "Flag": {1: "Uniform", 2: "Chebyshev First Kind", 3: "Chebyshev Second Kind"}[flag],
                "n": n,
                "Method": "Barycentric",
                "lambda_n": lambda_val,
                "h_n": h_val
            })

            # Newton interpolation:
            x_i_newton = ordering(np.copy(x_i), order_type)
            y_diff = newton_divided_diff(x_i_newton, y_i)
            p_newton = horners_rule(x_test, x_i_newton, y_diff)

            gamma = bary1_weights(x_i_newton)
            _, lambda_val, h_val = bary1_interpolation(x_test, x_i_newton, gamma, y_i)
            condition_table_newt.append({
                "Function": label,
                "Order": order,
                "Flag": {1: "Uniform", 2: "Chebyshev First Kind", 3: "Chebyshev Second Kind"}[flag],
                "n": n,
                "Method": "Barycentric",
                "lambda_n": lambda_val,
                "h_n": h_val
            })

            # Compute errors
            err_bary = relative_error(p_bary2, y_true)
            err_matrix_bary.append(err_bary)
            err_newt = relative_error(p_newton, y_true)
            err_matrix_newt.append(err_newt)

        # Plot errors for Barycentric interpolation
        plt.figure()
        plt.plot(x_test, err_matrix_bary[0], label="Uniform", color='blue', linestyle=':')
        plt.plot(x_test, err_matrix_bary[1], label="Chebyshev First Kind", color='red', linestyle=':')
        plt.plot(x_test, err_matrix_bary[2], label="Chebyshev Second Kind", color='green', linestyle=':')
        plt.legend()
        plt.grid()
        plt.xlabel("x")
        plt.ylabel("Relative Error")
        plt.title(f"Barycentric Interpolation Errors for {label}")
        plt.show()

        # Plot errors for Newton interpolation
        plt.figure()
        plt.plot(x_test, err_matrix_newt[0], label="Uniform", color='blue', linestyle=':')
        plt.plot(x_test, err_matrix_newt[1], label="Chebyshev First Kind", color='red', linestyle=':')
        plt.plot(x_test, err_matrix_newt[2], label="Chebyshev Second Kind", color='green', linestyle=':')
        plt.legend()
        plt.grid()
        plt.xlabel("x")
        plt.ylabel("Relative Error")
        plt.title(f"Newton Interpolation Errors for {label}, {order} order")
        plt.show()


'''------------------------ TESTER ------------------------'''
'''-------------------- FUNCTION f4(x) --------------------'''

# Define Testing Parameters
n_values = [20, 25, 29, 30, 31]
x_test = np.linspace(-1, 1, 100)  # Fine grid for error analysis
y_true = f4(x_test)
mesh_labels = {1: "Uniform", 2: "Chebyshev First Kind", 3: "Chebyshev Second Kind"}
colors = ["blue", "red", "green", "purple", "orange"]  # For different n values
order_types = {1: "Increasing", 2: "Decreasing", 3: "Leja"}

# Loop over mesh types (Uniform, Chebyshev 1st Kind, Chebyshev 2nd Kind)
for flag in [1, 2, 3]:
    plt.figure()
    for idx, n in enumerate(n_values):
        beta, x_i = bary2_weights(flag, n)
        y_i = f4(x_i)
        p_bary2 = bary2_interpolation(x_test, x_i, beta, y_i)
        err_bary = relative_error(p_bary2, y_true)
        plt.plot(x_test, err_bary, label=f"Barycentric (n={n})", linestyle=':', color=colors[idx])
    plt.xlabel("x")
    plt.ylabel("Relative Error")
    plt.yscale("log")  # Log scale for better visualization
    plt.title(f"Barycentric Interpolation Errors for {mesh_labels[flag]}")
    plt.legend()
    plt.grid()
    plt.show()

for flag in [1, 2, 3]:
    for order in order_types:
        plt.figure()
        for idx, n in enumerate(n_values):
            beta, x_i = bary2_weights(flag, n)
            x_i = ordering(x_i, order)
            y_i = f4(x_i)
            y_diff = newton_divided_diff(x_i, y_i)
            p_newton = horners_rule(x_test, x_i, y_diff)
            err_newt = relative_error(p_newton, y_true)
            plt.plot(x_test, err_newt, label=f"Newton (n={n})", linestyle=':', color=colors[idx])

        plt.xlabel("x")
        plt.ylabel("Relative Error")
        plt.yscale("log")
        plt.title(f"Newton Interpolation Errors for {mesh_labels[flag]}, {order_types[order]} order")
        plt.legend()
        plt.grid()
        plt.show()

