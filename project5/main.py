import numpy as np
import math

dtype = np.float64
#dtype = np.float32


# Functions f(x) for testing routines
def f1(x, d, rho=2, dtype=dtype):
    y = (x - rho) ** d
    y = np.asarray(y, dtype=dtype)
    return y

def f2(x, d, dtype=dtype):
    y = np.ones_like(x, dtype=dtype)
    for i in range(d + 1):
        y *= (x - i)
    return y

def f3(x, x_i, n, dtype=dtype):
    l = np.ones_like(x, dtype=dtype)
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
    l = np.ones(n, dtype=dtype)
    p_sum = np.zeros_like(x, dtype=dtype)
    w = np.ones_like(x, dtype=dtype)
    cond_num_1 = np.zeros_like(x, dtype=dtype)
    k_num = np.zeros_like(x, dtype=dtype)
    k_denom = np.zeros_like(x, dtype=dtype)
    exact = np.zeros_like(x, dtype=bool)  # Track exact matches

    for k in range(len(x)):
        for i in range(n):
            if x[k] == x_i[i]:  # Exact match found
                exact[k] = True
                p_sum[k] = y_i[i] # Directly assign function value
                w[k] = 1
            else:
                w[k] *= (x[k] - x_i[i])
                p_sum[k] += y_i[i] * gamma[i] / (x[k] - x_i[i])
            # Summation for numerator of interpolating condition number
            k_num[k] += np.abs(w[k] * y_i[i])
            k_denom[k] += w[k] * y_i[i]
        cond_num_1[k] += np.abs(w[k])

    p = np.zeros_like(x, dtype=dtype)  # Initialize p before assignment

    # Apply the formula only where no exact match was found
    mask = ~exact
    p[mask] = w[mask] * p_sum[mask]

    cond_num_y = k_num / np.abs(k_denom)

    print("barycentric 1 -- p(x):", p)
    print("barycentric 1 -- condition number k(x,n,y):", cond_num_y)
    print("barycentric 1 -- condition number k(x,n,1):", cond_num_1)
    return p, cond_num_y, cond_num_1


# Barycentric 2 form weights
def bary2_weights(flag, n, f, a=-1, b=1, dtype=dtype):
    beta = np.ones(n + 1, dtype=dtype)
    x_i = np.zeros(n + 1, dtype=dtype)

    if flag == 1:
        type = 'Uniform'
        x_i = np.linspace(a, b, n + 1, dtype=dtype)
        for i in range(n + 1):
            beta[i] = math.comb(n, i) * (-1) ** i

    elif flag == 2:
        type = 'Chebyshev First Kind'
        for i in range(n + 1):
            rad = ((2 * i) + 1) * np.pi / (2 * (n + 1))
            x_i[i] = math.cos(rad)
            beta[i] = math.sin(rad) * (-1) ** i

    elif flag == 3:
        type = 'Chebyshev Second Kind'
        for i in range(n + 1):
            rad = i * np.pi / n
            x_i[i] = math.cos(rad)
            beta[i] = 0.5 * (-1) ** i if i == 0 or i == n else (-1) ** i

    y_i = np.array([f(x) for x in x_i], dtype=dtype)

    print("barycentric 2 -- beta:", beta)
    print(f"barycentric 2 -- {type} x_i:", x_i)
    print("barycentric 2 -- f(x_i):", y_i)
    return beta, x_i, y_i


# Barycentric 2 form interpolating polynomial
def bary2_interpolation(x, x_i, beta, y_i):
    n = len(x_i)
    num = np.zeros_like(x, dtype=dtype)
    denom = np.zeros_like(x, dtype=dtype)
    p = np.zeros_like(x, dtype=dtype)
    exact = np.zeros_like(x, dtype=bool)  # Track exact matches

    for k in range(len(x)):
        for i in range(n):
            if x[k] == x_i[i]:  # Exact match found
                exact[k] = True
                p[k] = y_i[i]  # Directly assign function value
            else:
                term = beta[i] / (x[k] - x_i[i])
                num[k] += term * y_i[i]
                denom[k] += term

    # Compute p only where there was no exact match
    mask = ~exact  # Invert boolean mask to apply standard formula
    p[mask] = num[mask] / denom[mask]

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

    print("newton divided differences:", y_diff)
    return y_diff, y_i


def newton_interpolation(x, x_i, y_diff, dtype=dtype):
    n = len(y_diff)
    p = np.zeros_like(x, dtype=dtype)

    for k in range(len(x)):
        p[k] = y_diff[0]
        w = 1
        for i in range(1, n):
            w *= (x[k] - x_i[i - 1])
            p[k] += y_diff[i] * w

    print("newton interpolating polynomial -- p(x):", p)
    return p


# Horner's Rule
def horners_rule(x, x_i, y_diff):
    n = len(x_i)
    s = y_diff[-1] * np.ones_like(x, dtype=dtype)

    for k in range(len(x)):
        for i in range(n-1, -1, -1):
            s[k] = (s[k] * (x[k] - x_i[i])) + y_diff[i]

    print("horner's rule -- p(x):", s)
    return s


# Ordering Function
def ordering(x_i, flag):
    n = len(x_i)

    if flag == 1:
        type = 'Ascending'
        x_i.sort()

    elif flag == 2:
        type = 'Descending'
        x_i.sort(reverse=True)

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

    print(f"{type} ordering -- x_i:", x_i)
    return x_i


# Error Evaluation
def evaluate_p(p, f, x, dtype=dtype):
    r = p - f(x)
    print("p(x) - f(x) = r:", r)
    norm = np.linalg.norm(r, ord=np.inf)
    print("inf norm r:", norm)
    avg = np.mean(r)
    print("avg r:", avg)
    var = np.var(r)
    print("var r:", var)
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


# Function testing setup
x = np.array([-3., -2., -0.01, .25, .75, 3], dtype=dtype)

# Test functions f1(x), f2(x), f3(x)
test_functions = [
    (lambda x: f1(x, d=2), "Function 1 (f1)"),
    (lambda x: f2(x, d=3), "Function 2 (f2)"),
    (lambda x: f3(x, np.linspace(-1, 1, 4), 4), "Function 3 (f3)")  # f3 uses specific x_i
]

# Different node types for interpolation
node_types = [
    (1, "Uniform points"),
    (2, "Chebyshev First Kind"),
    (3, "Chebyshev Second Kind")
]

# Run tests for each function
for f, fname in test_functions:
    print(f"\nTesting {fname}:\n")

    for flag, node_type in node_types:
        print(f"\nUsing {node_type} for interpolation nodes\n")

        # Generate interpolation nodes
        beta, x_i, y_i = bary2_weights(flag, 9, f)

        # Barycentric 2 Interpolation
        print("\nBarycentric 2 interpolation")
        p_bary2 = bary2_interpolation(x, x_i, beta, y_i)
        evaluate_p(p_bary2, f, x)

        # Barycentric 1 Interpolation
        print("\nBarycentric 1 interpolation")
        gamma, _ = bary1_weights(x_i, f)
        p_bary1, cond_num_y, cond_num_1 = bary1_interpolation(x, x_i, gamma, y_i)
        evaluate_p(p_bary1, f, x)

        # Newton's Interpolation
        print("\nNewton Interpolation")
        y_diff, y_i_newton = newton_divided_diff(x_i, f)
        p_newton = newton_interpolation(x, x_i, y_diff)
        evaluate_p(p_newton, f, x)

        # Horner's Rule (Evaluation of Newton Polynomial)
        print("\nHorner's Rule Evaluation")
        p_horner = horners_rule(x, x_i, y_diff)
        evaluate_p(p_horner, f, x)

    print("\n" + "="*50 + "\n")


# Test function f4(x)
def test_interpolation(f, n_values, mesh_types):
    results = {}

    # Evaluation points (high resolution)
    x_eval = np.linspace(-1, 1, 100, dtype=dtype)

    for n in n_values:
        results[n] = {}

        for flag, mesh_name in mesh_types:
            print(f"\nTesting f4(x) with {mesh_name} (n={n})\n")

            # Generate interpolation nodes
            beta, x_i, y_i = bary2_weights(flag, n, f4)

            # Conditioning (Barycentric 1 Form)
            gamma, _ = bary1_weights(x_i, f4)
            _, cond_num_y, cond_num_1 = bary1_interpolation(x_eval, x_i, gamma, y_i)

            # Summarize conditioning
            cond_stats = summarize_conditioning(cond_num_y, cond_num_1)
            results[n][mesh_name] = cond_stats

            # Barycentric Form 2
            print("\nBarycentric 2 interpolation")
            p_bary2 = bary2_interpolation(x_eval, x_i, beta, y_i)
            evaluate_p(p_bary2, f4, x_eval)

            # Newton Interpolation (Increasing Order)
            print("\nNewton Interpolation (Increasing Order)")
            y_diff_inc, _ = newton_divided_diff(x_i, f4)
            p_newton_inc = newton_interpolation(x_eval, x_i, y_diff_inc)
            evaluate_p(p_newton_inc, f4, x_eval)

            # Newton Interpolation (Decreasing Order)
            print("\nNewton Interpolation (Decreasing Order)")
            x_i_dec = np.flip(x_i)  # Reverse order
            y_diff_dec, _ = newton_divided_diff(x_i_dec, f4)
            p_newton_dec = newton_interpolation(x_eval, x_i_dec, y_diff_dec)
            evaluate_p(p_newton_dec, f4, x_eval)

            # Newton Interpolation (Leja Ordering)
            print("\nNewton Interpolation (Leja Ordering)")
            x_i_leja = ordering(x_i, flag=3)
            y_diff_leja, _ = newton_divided_diff(x_i_leja, f4)
            p_newton_leja = newton_interpolation(x_eval, x_i_leja, y_diff_leja)
            evaluate_p(p_newton_leja, f4, x_eval)

    return results

# Define n values and interpolation meshes
n_values = [5, 10, 15, 20]  # Different values of n
mesh_types = [
    (1, "Uniform Points"),
    (2, "Chebyshev First Kind"),
    (3, "Chebyshev Second Kind")
]

# Run tests for f4(x)
def summarize_conditioning(cond_num_y, cond_num_1):
    Λn = np.max(cond_num_y)  # Max condition number
    Hn = np.mean(cond_num_y)  # Average condition number
    stats = {
        "Λn (max cond num)": Λn,
        "Hn (avg cond num)": Hn,
        "Min κ(x, n, y)": np.min(cond_num_y),
        "Max κ(x, n, y)": np.max(cond_num_y),
        "Var κ(x, n, y)": np.var(cond_num_y),
        "Min κ(x, n, 1)": np.min(cond_num_1),
        "Max κ(x, n, 1)": np.max(cond_num_1),
        "Var κ(x, n, 1)": np.var(cond_num_1)
    }
    return stats


conditioning_results = test_interpolation(f4, n_values, mesh_types)

# Display summarized conditioning results
for n, meshes in conditioning_results.items():
    print(f"\nSummary for n={n}")
    for mesh, stats in meshes.items():
        print(f"\n{mesh}:")
        for key, value in stats.items():
            print(f"{key}: {value:.5f}")