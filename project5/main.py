import numpy as np
import math

dtype = np.float64


# Functions f(x) for testing routines
def f1(x, d, dtype=dtype):
    x = np.asarray(x, dtype=dtype)
    return (x - 2) ** d


def f2(x, d, dtype=dtype):
    x = np.asarray(x, dtype=dtype)
    y = np.ones_like(x, dtype=dtype)
    for i in range(d + 1):
        y *= (x - i)
    return y


def f3(x, x_i, n, dtype=dtype):
    x = np.asarray(x, dtype=dtype)
    x_i = np.asarray(x_i, dtype=dtype)
    l = np.ones_like(x, dtype=dtype)
    for i in range(n):
        for j in range(n):
            if i != j:
                l *= (x - x_i[j]) / (x_i[i] - x_i[j])
    return l


def f4(x, dtype=dtype):
    x = np.asarray(x, dtype=dtype)
    return 1 / (1 + (25 * x ** 2))


# Barycentric 1 form weights
def bary1_weights(x_i, f, dtype=dtype):
    n = len(x_i)
    gamma = np.ones(n, dtype=dtype)
    y_i = np.array([f(x) for x in x_i], dtype=dtype)

    for i in range(n):
        for j in range(n):
            if i != j:
                gamma[i] /= (x_i[i] - x_i[j])

    print("bary1_weights -> gamma:", gamma)
    print("bary1_weights -> y_values:", y_i)
    return gamma, y_i


# Barycentric 1 form interpolating polynomial
def bary1_interpolation(x, x_i, gamma, y_i, dtype=dtype):
    n = len(x_i)
    l = np.ones(n, dtype=dtype)
    p_sum = np.zeros_like(x, dtype=dtype)
    w = np.ones_like(x, dtype=dtype)
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
            for j in range(i):
                if i != j:
                    l[i] *= (x_i[i] - x_i[j])
            k_num[k] += np.abs(l[i] * y_i[i])
            k_denom[k] += l[i] * y_i[i]

    p = np.zeros_like(x, dtype=dtype)  # Initialize p before assignment

    # Apply the formula only where no exact match was found
    mask = ~exact
    p[mask] = w[mask] * p_sum[mask]

    cond_num = k_num / k_denom

    print("bary1_interpolation -> p:", p)
    print("bary1_interpolation -> cond_num:", cond_num)
    return p, cond_num


# Barycentric 2 form weights
def bary2_weights(flag, n, f, a=-1, b=1, dtype=dtype):
    beta = np.ones(n + 1, dtype=dtype)
    x_i = np.zeros(n + 1, dtype=dtype)

    if flag == 1:
        x_i = np.linspace(a, b, n + 1, dtype=dtype)
        for i in range(n + 1):
            beta[i] = math.comb(n, i) * (-1) ** i

    elif flag == 2:
        for i in range(n + 1):
            rad = ((2 * i) + 1) * np.pi / (2 * (n + 1))
            x_i[i] = math.cos(rad)
            beta[i] = math.sin(rad) * (-1) ** i

    elif flag == 3:
        for i in range(n + 1):
            rad = i * np.pi / n
            x_i[i] = math.cos(rad)
            beta[i] = 0.5 * (-1) ** i if i == 0 or i == n else (-1) ** i

    y_i = np.array([f(x) for x in x_i], dtype=dtype)

    print("bary2_weights -> beta:", beta)
    print("bary2_weights -> x_i:", x_i)
    print("bary2_weights -> y_i:", y_i)
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

    print("bary2_interpolation -> p:", p)
    return p


# Newton Divided Differences
def newton_divided_diff(x_i, f, dtype=dtype):
    n = len(x_i)
    y_i = np.array([f(x) for x in x_i], dtype=dtype)
    y_diff = np.copy(y_i)

    for k in range(1, n):
        for i in range(n-k):
            # This is the right side of the divided differences table
            y_diff[i] = (y_diff[i + 1] - y_diff[i]) / (x_i[i + k] - x_i[i])


    print("newton_divided_diff -> y_diff:", y_diff)
    print("newton_divided_diff -> y_i:", y_i)
    return y_diff, y_i


# Horner's Rule
def horners_rule(x, x_i, y_diff):
    n = len(x_i)
    s = y_diff[0] * np.ones_like(x, dtype=dtype)

    for k in range(len(x)):
        for i in range(n):
            s[k] = s[k] * (x[k] - x_i[i]) + y_diff[i]

    print("horners_rule -> s:", s)
    return s


# Ordering Function
def ordering(x_i, flag):
    n = len(x_i)
    if flag == 1:
        x_i.sort()
    elif flag == 2:
        x_i.sort(reverse=True)
    elif flag == 3:
        leja_points = [x_i[0]]
        remaining_points = list(x_i[1:])

        for _ in range(n - 1):
            next_point = max(remaining_points, key=lambda x: np.prod(np.abs(x - np.array(leja_points))))
            leja_points.append(next_point)
            remaining_points.remove(next_point)

        x_i = np.array(leja_points)

    print("ordering -> x_values:", x_i)
    return x_i

# Error Evaluation
def evaluate_p(p, f, x, dtype=dtype):
    r = p - f(x)
    print("evaluate_p -> r:", r)
    norm = np.linalg.norm(r, ord=np.inf)
    print("evaluate_p -> norm:", norm)
    avg = np.mean(r)
    print("evaluate_p -> avg:", avg)
    var = np.var(r)
    print("evaluate_p -> var:", var)
    return norm, avg, var

# Sample Points
# x_i values to create interpolating polynomial
x_i = np.asarray([0, 0.5, 1], dtype=dtype)
# x values to test interpolating polynomial
x = np.asarray([-1, .75, 2, 3], dtype=dtype)

# Sample Function
def f5(x): return np.asarray((6 * x) + 2, dtype=dtype)

gamma, y_i = bary1_weights(x_i, f5)
p, cond_num = bary1_interpolation(x, x_i, gamma, y_i)

beta, x_i2, y_i = bary2_weights(3, 6, f5)
x_i3 = ordering(x_i2, flag=3)
p2 = bary2_interpolation(x, x_i2, beta, y_i)

y_diff, y_i = newton_divided_diff(x_i, f5)
s = horners_rule(x, x_i, y_diff)

norm, avg, var = evaluate_p(p, f5, x)
norm2, avg2, var2 = evaluate_p(p2, f5, x)

