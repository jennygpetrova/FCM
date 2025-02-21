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


def f3(x, x_i, dtype=dtype):
    # This version computes the 0th Lagrange basis function.
    l = np.ones_like(x, dtype=dtype)
    n = len(x_i)
    for j in range(1, n):
        l *= (x - x_i[j]) / (x_i[0] - x_i[j])
    return l


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
    p_sum = np.zeros_like(x, dtype=dtype)
    w = np.ones_like(x, dtype=dtype)
    k_num = np.zeros_like(x, dtype=dtype)
    k_denom = np.zeros_like(x, dtype=dtype)
    kappa_1_array = np.zeros_like(x, dtype=dtype)

    for k in range(len(x)):
        for i in range(n):
            # If x[k] is very close to any node, we directly use that y_i.
            if np.isclose(x[k], x_i, rtol=1e-05, atol=1e-08).any():
                p_sum[k] = y_i[i]
                w[k] = 1
            else:
                w[k] *= (x[k] - x_i[i])
                p_sum[k] += y_i[i] * gamma[i] / (x[k] - x_i[i])
            k_num[k] += np.abs(w[k] * y_i[i])
            k_denom[k] += w[k] * y_i[i]
        if np.abs(w[k]) < 1:
            kappa_1_array[k] = 1
        else:
            kappa_1_array[k] = np.abs(w[k])
    lambda_n_scalar = np.max(kappa_1_array)
    p = w * p_sum
    kappa_y = k_num / np.abs(k_denom)
    print("barycentric 1 -- p(x):", p)
    return p, kappa_y, lambda_n_scalar


# Barycentric 2 form weights and interpolation
def bary2_weights(flag, n, f, a=-1, b=1, dtype=dtype):
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

    # For f3 (which requires an extra parameter), we skip computing y_i here.
    if f.__name__ == "f3":
        y_i = np.empty_like(x_i)
    else:
        y_i = np.array([f(x) for x in x_i], dtype=dtype)
    print("barycentric 2 -- f(x_i):", y_i)
    return beta, x_i, y_i


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
def newton_divided_diff(x_i, f, dtype=dtype):
    n = len(x_i)
    # f is expected to accept a single argument.
    y_i = np.array([f(x) for x in x_i], dtype=dtype)
    y_diff = np.zeros_like(x_i, dtype=dtype)
    y_diff[0] = y_i[0]
    for k in range(1, n):
        for i in range(n - k):
            y_i[i] = (y_i[i + 1] - y_i[i]) / (x_i[i + k] - x_i[i])
        y_diff[k] = y_i[0]
    return y_diff, y_i


# Vectorized Horner's Rule for Newton interpolation
def horners_rule(x, x_i, y_diff):
    x = np.asarray(x, dtype=dtype)  # ensure x is a NumPy array
    s = np.full(x.shape, y_diff[-1], dtype=dtype)
    for i in range(len(x_i) - 2, -1, -1):
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
'''------------- FUNCTIONS f1(x) f2(x) f3(x) -------------'''
n = 29
eps = np.finfo(float).eps
x_test = np.linspace(0 + (10 ** 3 * eps), 0.75 - (10 ** 3 * eps), 30)

# Define our functions and labels.
# For f2 we fix the degree to d=9 using a lambda.
functions = [
    f1,
    lambda x: f2(x, d=9),
    f3  # f3 requires the node set, so we handle it specially.
]
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
        err_matrix_bary = []
        err_matrix_newt = []
        for flag in [1, 2, 3]:
            # Barycentric interpolation
            beta, x_i, y_i = bary2_weights(flag, n, f)
            if label == "Lagrange Basis Product":
                y_i = f3(x_i, x_i, dtype=dtype)
            p_bary2 = bary2_interpolation(x_test, x_i, beta, y_i)
            if label == "Lagrange Basis Product":
                y_true = f3(x_test, x_i, dtype=dtype)
            else:
                y_true = f(x_test)

            # Newton interpolation:
            x_i = ordering(x_i, order_type)
            # For f3, wrap it in a lambda that supplies x_i.
            if label == "Lagrange Basis Product":
                new_f = lambda x: f3(x, x_i, dtype=dtype)
            else:
                new_f = f
            y_diff, _ = newton_divided_diff(x_i, new_f, dtype=dtype)
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



# print("------------------------------------------------------------")
#
# # Test function f4(x)
# print("\nTesting f4:\n")
# n_values = [5, 10, 15, 20, 25]
# for flag, type in mesh_types:
#     for n in n_values:
#         print(f"\nUsing {type} for interpolation nodes\n")
#
#         # Barycentric 2 Interpolation
#         print("\nBarycentric 2 interpolation")
#         print(f"\nDegree n = {n}")
#         beta, x_i, y_i = bary2_weights(flag, n, f)
#         p_bary2 = bary2_interpolation(x, x_i, beta, y_i)
#         norm, avg, var = evaluate_p(p_bary2, f, x)
#
#         # Newton's Interpolation
#         print("\nNewton Interpolation")
#         y_diff = newton_divided_diff(x_i, f)[0]
#         print("\nNewton Interpolation for increasing order mesh points")
#         x1 = ordering(x, 1)
#         norm1, avg1, var1 = evaluate_p(p_newton, f, x1)
#         print("\nNewton Interpolation for decreasing order mesh points")
#         x2 = ordering(x, 2)
#         norm2, avg2, var2 = evaluate_p(p_newton, f, x2)
#         print("\nNewton Interpolation for Leja ordering mesh points")
#         x3 = ordering(x, 3)
#         norm3, avg3, var3 = evaluate_p(p_newton, f, x3)





# # Tester for functions 1-3
# conditions = {}
# for f, fname in test_functions:
#     for flag in mesh_types:
#         beta, x_i, y_i = bary2_weights(flag, 9, f)
#         p_bary2 = bary2_interpolation(x_eval, x_i, beta, y_i)
#         gamma = bary1_weights(x_i, f)[0]
#         p_bary1, cond_num_y, cond_num_1 = bary1_interpolation(x_eval, x_i, gamma, y_i)
#         norm, avg, var = evaluate_p(p_bary2, f, x_eval)
#         conditions[f"{fname}-flag{flag}"] = cond_num_y, cond_num_1
#
#         plt.figure()
#         plt.plot(x_eval, f(x_eval), label=f'{fname}(x)', linestyle='dashed')
#         plt.plot(x_eval, p_bary2, label=f'Interpolant {fname} (flag {flag})', alpha=0.7)
#         plt.scatter(x_i, y_i, color='red', zorder=3)
#         plt.xlabel('x')
#         plt.ylabel('y')
#         plt.legend()
#         plt.title(f'Interpolation for {fname} (flag {flag})')
#         plt.show()
#
#
# # Investigate convergence behavior for f4
# plt.figure(figsize=(10, 6))
# for flag in mesh_types:
#     errors = []
#     for n in range(5, 51, 5):
#         beta, x_i, y_i = bary2_weights(flag, n, f4)
#         p_bary2 = bary2_interpolation(x_eval, x_i, beta, y_i)
#         norm, avg, var = evaluate_p(p_bary2, f, x_eval)
#     plt.plot(range(5, 51, 5), norm, marker='o', label=f'flag {flag}')
#
# plt.xlabel('n')
# plt.ylabel('Infinity Norm of Error')
# plt.title('Convergence of f4 Approximation for Different Mesh Types')
# plt.legend()
# plt.grid()
# plt.show()






