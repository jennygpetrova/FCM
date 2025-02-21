import numpy as np
import matplotlib.pyplot as plt
import math

dtype = np.float64
#dtype = np.float32


# Functions f(x) for testing routines
def f1(x, d=2, rho=2, dtype=dtype):
    y = (x - rho) ** d
    y = np.asarray(y, dtype=dtype)
    return y

def f2(x, d, dtype=dtype):
    y = np.ones_like(x, dtype=dtype)
    for i in range(d + 1):
        y *= (x - i)
    return y

def f3(x, x_i, dtype=dtype):
    l = np.ones_like(x, dtype=dtype)
    n = len(x_i)
    for i in range(n):
        for j in range(n):
            if i != j:
                l *= (x - x_i[j]) / (x_i[i] - x_i[j])
    return l

def f4(x, dtype=dtype):
    y = 1 / (1 + (25 * x ** 2))
    y = np.asarray(y, dtype=dtype)
    return y


# Barycentric 1 form weights
def bary1_weights(x_i, f, dtype=dtype):
    n = len(x_i)
    gamma = np.ones(n, dtype=dtype)
    y_i = np.array([f(x) for x in x_i], dtype=dtype)

    for i in range(n):
        for j in range(n):
            if i != j:
                gamma[i] /= (x_i[i] - x_i[j])

    print("barycentric 1 -- f(x_i):", y_i)
    return gamma, y_i


# Barycentric 1 form interpolating polynomial
def bary1_interpolation(x, x_i, gamma, y_i, dtype=dtype):
    n = len(x_i)
    p_sum = np.zeros_like(x, dtype=dtype)
    w = np.ones_like(x, dtype=dtype)
    kappa_1 = np.zeros_like(x, dtype=dtype)
    lambda_n = np.ones_like(x, dtype=dtype)
    k_num = np.zeros_like(x, dtype=dtype)
    k_denom = np.zeros_like(x, dtype=dtype)

    for k in range(len(x)):
        for i in range(n):
            if np.isclose(x[k], x_i, rtol=1e-05, atol=1e-08, equal_nan=False).any():
                p_sum[k] = y_i[i] # Directly assign function value
            else:
                w[k] *= (x[k] - x_i[i])
                p_sum[k] += y_i[i] * gamma[i] / (x[k] - x_i[i])
            # Summation for numerator of interpolating condition number
            k_num[k] += np.abs(w[k] * y_i[i])
            k_denom[k] += w[k] * y_i[i]
        lambda_n[k] = np.max(w)
        kappa_1[k] += np.abs(w[k])

    p = w * p_sum

    kappa_y = k_num / np.abs(k_denom)

    print("barycentric 1 -- p(x):", p)
    # print("barycentric 1 -- condition number k(x,n,y):", cond_num_y)
    # print("barycentric 1 -- condition number k(x,n,1):", kappa_1)
    return p, kappa_y, kappa_1, lambda_n


# Barycentric 2 form weights
def bary2_weights(flag, n, f, a=-1, b=1, dtype=dtype):
    beta = np.ones(n + 1, dtype=dtype)
    x_i = np.zeros(n + 1, dtype=dtype)

    if flag == 1:
        # type = 'Uniform'
        x_i = np.linspace(a, b, n + 1, dtype=dtype)
        beta[0] = 1
        for i in range(1, n + 1):
            beta[i] = beta[i-1] * (-1)*(n-i+1)/(i)

    elif flag == 2:
        # type = 'Chebyshev First Kind'
        for i in range(n + 1):
            rad = ((2 * i) + 1) * np.pi / (2 * (n + 1))
            x_i[i] = math.cos(rad)
            beta[i] = math.sin(rad) * (-1) ** i

    elif flag == 3:
        # type = 'Chebyshev Second Kind'
        for i in range(n + 1):
            rad = i * np.pi / n
            x_i[i] = math.cos(rad)
            beta[i] = 0.5 * (-1) ** i if i == 0 or i == n else (-1) ** i

    y_i = np.array([f(x) for x in x_i], dtype=dtype)

    # print("barycentric 2 -- beta:", beta)
    # print(f"barycentric 2 -- {type} x_i:", x_i)
    print("barycentric 2 -- f(x_i):", y_i)
    return beta, x_i, y_i


# Barycentric 2 form interpolating polynomial
def bary2_interpolation(x, x_i, beta, y_i):
    n = len(x_i)
    num = np.zeros_like(x, dtype=np.float64)
    denom = np.zeros_like(x, dtype=np.float64)
    p = np.zeros_like(x, dtype=np.float64)

    for k in range(len(x)):
        for i in range(n):  # Ensure 'i' does not exceed 'n-1'
            if i >= len(beta) or i >= len(y_i):  # Safety check
                print(f"Index out of bounds: i={i}, len(beta)={len(beta)}, len(y_i)={len(y_i)}")
                continue  # Skip iteration if an index mismatch is found

            if np.isclose(x[k], x_i[i]):  # Properly check for exact match
                p[k] = y_i[i]  # Assign function value directly
                break
            else:
                term = beta[i] / (x[k] - x_i[i])  # Safe access
                num[k] += term * y_i[i]
                denom[k] += term

        if denom[k] != 0:  # Avoid division by zero
            p[k] = num[k] / denom[k]

    print("barycentric 2 -- p(x):", p)
    return p



# Newton Divided Differences
def newton_divided_diff(x_i, f, dtype=dtype):
    n = len(x_i)
    y_i = np.array([f(x) for x in x_i], dtype=dtype)
    y_diff = np.zeros_like(x_i, dtype=dtype)

    y_diff[0] = y_i[0]

    for k in range(1, n):
        for i in range(n - k):  # Compute the k-th level divided difference
            y_i[i] = (y_i[i + 1] - y_i[i]) / (x_i[i + k] - x_i[i])  # Use recursive formula
        y_diff[k] = y_i[0]

    # print("newton divided differences:", y_diff)
    return y_diff, y_i


# def newton_interpolation(x, x_i, y_diff, dtype=dtype):
#     n = len(y_diff)
#     p = np.zeros_like(x, dtype=dtype)
#
#     for k in range(len(x)):
#         p[k] = y_diff[0]
#         w = 1
#         for i in range(1, n):
#             w *= (x[k] - x_i[i - 1])
#             p[k] += y_diff[i] * w
#
#     # print("newton interpolating polynomial -- p(x):", p)
#     return p


# Horner's Rule
def horners_rule(x, x_i, y_diff):
    n = len(x_i)
    s = y_diff[-1] * np.ones_like(x, dtype=dtype)

    for k in range(len(x)):
        for i in range(n-1, -1, -1):
            s[k] = (s[k] * (x[k] - x_i[i])) + y_diff[i]

    # print("horner's rule -- p(x):", s)
    return s


# Ordering Function
def ordering(x_i, flag):
    if flag == 1:
        type = 'Ascending'
        x_i.sort()

    elif flag == 2:
        type = 'Descending'
        x_i.sort()
        x_i = x_i[::-1]

    elif flag == 3:
        type = 'Leja'
        x_remaining = np.copy(x_i)
        x_leja = [x_remaining[0]]
        x_remaining = np.delete(x_remaining, 0)

        while len(x_remaining) > 0:
            # Compute product of absolute differences
            for x in x_remaining:
                prod = np.array([np.prod(np.abs(x - np.array(x_leja)))])

            index = np.argmax(prod)
            x_leja.append(x_remaining[index])
            x_remaining = np.delete(x_remaining, index)

        x_i = np.array(x_leja)

    # print(f"{type} ordering -- x_i:", x_i)
    return x_i


# Error Evaluation
def evaluate_p(p, f, x, dtype=dtype):
    r = p - f(x)
    #print("p(x) - f(x) = r:", r)
    norm = np.linalg.norm(r, ord=np.inf)
    #print("inf norm r:", norm)
    avg = np.mean(r)
    #print("avg r:", avg)
    var = np.var(r)
    #print("var r:", var)
    return norm, avg, var


# Sample points (Problem 2.1 in Study Set 2) to assess accuracy while building code

# # x_i values to create interpolating polynomial
# x_i = np.asarray([0, 0.5, 1], dtype=dtype)
#
# # x values to test interpolating polynomial
# x = np.asarray([-1, .75, 2, 3], dtype=dtype)
#
# # Sample function (known interpolating polynomial for the y_i values given in Set 2)
# def f5(x): return np.asarray((6 * x) + 2, dtype=dtype)
#
# gamma, y_i = bary1_weights(x_i, f5)
# p, cond_num_y, cond_num_1 = bary1_interpolation(x, x_i, gamma, y_i)
#
# beta, x_i2, y_i = bary2_weights(3, 6, f5)
# x_i3 = ordering(x, flag=3)
# p2 = bary2_interpolation(x, x_i2, beta, y_i)
#
# y_diff, y_i = newton_divided_diff(x_i, f5)
# p3 = newton_interpolation(x, x_i, y_diff)
# s = horners_rule(x, x_i, y_diff)
#
# norm, avg, var = evaluate_p(p, f5, x)




'''------------------------TESTER------------------------'''
# Run tests for function f1, f2, f3
def relative_error(p, f):
    n = len(f)
    err = []
    for i in range(n):
        err.append(np.abs(p[i] - f[i]) / f[i])
    return err

n = 29
eps = np.finfo(float).eps
x_test = np.linspace(-1 + (10**3 * eps), 1 - (10**3 * eps), 100)


# Uniform Mesh
beta, x_i, y_i = bary2_weights(1, n, f1)
p_bary2 = bary2_interpolation(x_test, x_i, beta, y_i)
norm, avg, var = evaluate_p(p_bary2, f1, x_test)
gamma = bary1_weights(x_i, f1)[0]
p_bary1, kappa_y, kappa_1, lambda_n = bary1_interpolation(x_test, x_i, gamma, y_i)

# Chebyshev First Kind Mesh
beta1, x_i1, y_i1 = bary2_weights(2, n, f1)
p_bary2_1 = bary2_interpolation(x_test, x_i1, beta, y_i1)
norm1, avg1, var1 = evaluate_p(p_bary2_1, f1, x_test)
gamma1 = bary1_weights(x_i, f1)[0]
p_bary1_1, kappa_y_1, kappa_1_1, lambda_n_1 = bary1_interpolation(x_test, x_i1, gamma, y_i1)

#Chebyshev Second Kind Mesh


# Compute errors
x_test = x_test[:-1]
y = np.array([f1(x) for x in x_test], dtype=dtype)
err_bary_uni = np.array(relative_error(p_bary2, y), dtype=dtype)
err_bary_cheb1 = np.array(relative_error(p_bary2_1, y), dtype=dtype)

# Plot
plt.plot(x_test, err_bary_uni, label="Barycentric (Uniform)", color='blue')
plt.plot(x_test, err_bary_cheb1, label="Barycentric (Chebychev)", color='red')
plt.legend()
plt.xlabel("x")
plt.ylabel("Relative Error")
plt.title("Interpolation Errors")
plt.show()




# for f, fname in test_functions:
#     print(f"\nTesting {fname}:\n")
#
#     for flag, type in mesh_types:
#         print(f"\nUsing {type} for interpolation nodes\n")
#
#         # Barycentric 2 Interpolation
#         print("\nBarycentric 2 interpolation")
#         # Test different degree types (n) for f2 and f3
#         n = 9 # default
#         # n = 21
#         # n = 29
#         print(f"\nDegree n = {n}")
#         beta, x_i, y_i = bary2_weights(flag, n, f)
#         p_bary2 = bary2_interpolation(x, x_i, beta, y_i)
#         evaluate_p(p_bary2, f, x)
#
#
#         # Barycentric 1 Interpolation
#         print("\nBarycentric 1 interpolation")
#         gamma = bary1_weights(x_i, f)[0]
#         p_bary1, cond_num_y, cond_num_1 = bary1_interpolation(x, x_i, gamma, y_i)
#         norm1, avg1, var1 = evaluate_p(p_bary1, f, x)
#
#         # Newton's Interpolation
#         print("\nNewton Interpolation")
#         y_diff, y_i_newton = newton_divided_diff(x_i, f)
#         print("\nNewton Interpolation for increasing order mesh points")
#         x1 = ordering(x, 1)
#         norm2, avg2, var2 = evaluate_p(p_newton, f, x1)
#         print("\nNewton Interpolation for decreasing order mesh points")
#         x2 = ordering(x, 2)
#         norm3, avg3, var3 = evaluate_p(p_newton, f, x2)
#         print("\nNewton Interpolation for Leja ordering mesh points")
#         x3 = ordering(x, 3)
#         norm4, avg4, var4 = evaluate_p(p_newton, f, x3)
#
#         # Horner's Rule (Evaluation of Newton Polynomial)
#         print("\nHorner's Rule Evaluation")
#         p_horner = horners_rule(x, x_i, y_diff)
#         norm5, avg5, var5 = evaluate_p(p_horner, f, x)








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






