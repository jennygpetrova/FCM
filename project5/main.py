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


def f3(x, x_values, n, dtype=dtype):
    x = np.asarray(x, dtype=dtype)
    x_values = np.asarray(x_values, dtype=dtype)
    l = np.ones_like(x, dtype=dtype)
    for i in range(n):
        for j in range(n):
            if i != j:
                l *= (x - x_values[j]) / (x_values[i] - x_values[j])
    return l


def f4(x, dtype=dtype):
    x = np.asarray(x, dtype=dtype)
    return 1 / (1 + (25 * x ** 2))


# Barycentric 1 form weights
def bary1_weights(x_values, f, dtype=dtype):
    n = len(x_values)
    gamma = np.ones(n, dtype=dtype)
    y_values = np.array([f(x) for x in x_values], dtype=dtype)

    for i in range(n):
        for j in range(n):
            if i != j:
                gamma[i] /= (x_values[i] - x_values[j])

    print("bary1_weights -> gamma:", gamma)
    print("bary1_weights -> y_values:", y_values)
    return gamma, y_values


# Barycentric 1 form interpolating polynomial
def bary1_interpolation(x, x_values, gamma, y_values, dtype=dtype):
    x = np.asarray(x, dtype=dtype)
    n = len(x_values)
    w = np.prod(x - x_values)
    sum = 0

    for i in range(n):
        sum += y_values[i] * gamma[i] / (x - x_values[i])

    p = np.asarray(w * sum, dtype=dtype)
    print("bary1_interpolation -> p:", p)
    return p


# Barycentric 2 form weights
def bary2_weights(flag, n, f, dtype=dtype):
    beta = np.ones(n + 1, dtype=dtype)
    x_values = np.zeros(n + 1, dtype=dtype)
    a, b = -1, 1

    if flag == 1:
        x_values = np.linspace(a, b, n + 1, dtype=dtype)
        for i in range(n + 1):
            beta[i] = math.comb(n, i) * (-1) ** i

    elif flag == 2:
        for i in range(n + 1):
            rad = ((2 * i) + 1) * np.pi / (2 * (n + 1))
            x_values[i] = math.cos(rad)
            beta[i] = math.sin(rad) * (-1) ** i

    elif flag == 3:
        for i in range(n + 1):
            rad = i * np.pi / n
            x_values[i] = math.cos(rad)
            beta[i] = 0.5 * (-1) ** i if i == 0 or i == n else (-1) ** i

    y_values = np.array([f(x) for x in x_values], dtype=dtype)

    print("bary2_weights -> beta:", beta)
    print("bary2_weights -> x_values:", x_values)
    print("bary2_weights -> y_values:", y_values)
    return beta, x_values, y_values


# Barycentric 2 form interpolating polynomial
def bary2_interpolation(x, x_values, beta, y_values):
    n = len(x_values)
    num = 0
    denom = 0

    for i in range(n):
        num += beta[i] * y_values[i] / (x - x_values[i])
        denom += beta[i] / (x - x_values[i])

    p = num / denom
    print("bary2_interpolation -> p:", p)
    return p


# Newton Divided Differences
def newton_divided_diff(x_values, f, dtype=dtype):
    n = len(x_values)
    y_values = np.array([f(x) for x in x_values], dtype=dtype)
    y_diff = np.copy(y_values)

    for k in range(1, n):
        for i in range(n - k):
            # This is the right side of the divided differences table
            y_diff[i] = (y_diff[i + 1] - y_diff[i]) / (x_values[i + k] - x_values[i])

    print("newton_divided_diff -> y_diff:", y_diff)
    print("newton_divided_diff -> y_values:", y_values)
    return y_diff, y_values


# Horner's Rule
def horners_rule(x, x_values, y_diff):
    n = len(x_values)
    s = y_diff[n - 1]
    for i in range(n - 2, -1, -1):
        s = s * (x - x_values[i]) + y_diff[i]

    print("horners_rule -> p:", s)
    return s


# Ordering Function
def ordering(x_values, flag):
    n = len(x_values)
    if flag == 1:
        x_values.sort()
    elif flag == 2:
        x_values.sort(reverse=True)
    elif flag == 3:
        leja_points = [x_values[0]]
        remaining_points = list(x_values[1:])

        for _ in range(n - 1):
            next_point = max(remaining_points, key=lambda x: np.prod(np.abs(x - np.array(leja_points))))
            leja_points.append(next_point)
            remaining_points.remove(next_point)

        x_values = np.array(leja_points)

    print("ordering -> x_values:", x_values)
    return x_values


# Error Evaluation
def evaluate_p(p, f, dtype=dtype):
    r = np.asarray(p - f, dtype=dtype)
    print("evaluate_p -> r:", r)
    return np.linalg.norm(r, ord=np.inf), np.average(r), np.var(r)

# Sample Tests
x_values = np.asarray([0, 0.5, 1], dtype=dtype)

def f5(x): return np.asarray((6 * x) + 2, dtype=dtype)

gamma, y_values = bary1_weights(x_values, f5)
p = bary1_interpolation(2, x_values, gamma, y_values)

beta, x_values2, y_values = bary2_weights(3, 6, f5)
x_values3 = ordering(x_values2, flag=3)
p2 = bary2_interpolation(2, x_values2, beta, y_values)

y_diff, y_values = newton_divided_diff(x_values, f5)
s = horners_rule(2, x_values, y_diff)

