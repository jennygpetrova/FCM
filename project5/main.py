import numpy as np
import math

dtype=np.float64

# Functions f(x) for testing routines
def f1(x, d, dtype=dtype):
    x = np.asarray(x, dtype=dtype)
    y = (x-2) ** d
    return y

def f2(x, d, dtype=dtype):
    x = np.asarray(x, dtype=dtype)
    y = np.ones_like(x, dtype=dtype)
    for i in range(d+1):
        y *= (x - i)
    return y

def f3(x, x_values, n, dtype=dtype):
    x = np.asarray(x, dtype=dtype)
    x_values = np.asarray(x_values, dtype=dtype)
    for i in range(n):
        l = np.asarray(1, dtype=dtype)
        for j in range(n):
            if i != j:
                l *= (x - x_values[j]) / (x_values[i] - x_values[j])
    return l

def f4(x, dtype=dtype):
    x = np.asarray(x, dtype=dtype)
    y = 1 / (1 + (25 * x ** 2))
    return y


# Barycentric 1 form weights
def bary1_weights(x_values, f, dtype=dtype):
    x_values = np.asarray(x_values, dtype=dtype)
    n = len(x_values)
    gamma = np.ones(n, dtype=dtype)
    y_values = np.array([f(x) for x in x_values], dtype=dtype)

    for i in range(n):
        for j in range(n):
            if i != j:
                gamma[i] /= (x_values[i] - x_values[j])

    return gamma, y_values

# Barycentric 1 form interpolating polynomial
def bary1_interpolation(x, x_values, gamma, y_values, dtype=dtype):
    x = np.asarray(x, dtype=dtype)
    x_values = np.asarray(x_values, dtype=dtype)
    n = len(x_values)
    sum = 0 # Integers automatically converted to floats for computations

    for i in range(n):
        w = 1
        for j in range(n):
            w *= (x - x_values[j])
        sum += y_values[i] * gamma[i] / (x - x_values[i])

    p = np.asarray(w * sum, dtype=dtype)
    return p


# Barycentric 2 form weights
def bary2_weights(flag, n, f, dtype=dtype):
    beta = np.ones(n, dtype=dtype)
    x_values = np.asarray(n, dtype=dtype)
    a = -1
    b = 1

    # Uniform points
    if flag == 1:
        x_values = np.linspace(a, b, n, dtype=dtype)
        for i in range(n+1):
            binom_coeff = math.factorial(n) / (math.factorial(i) * math.factorial(n - i))
            beta[i] = binom_coeff * (-1)**i

    # Chebyshev points of the first kind
    elif flag == 2:
        for i in range(n+1):
            rad = ((2*i) + 1) * np.pi / ((2*i) + 1)
            x_values[i] = math.cos(rad)
            beta[i] = math.sin(rad) * (-1)**i

    # Chebyshev points of the second kind
    elif flag == 3:
        for i in range(n+1):
            rad = i * np.pi / n
            x_values[i] = math.cos(rad)
            delta = 1
            if i == 0 or i == n:
                delta = .5
            beta[i] = delta * (-1)**i

    y_values = np.array([f(x) for x in x_values], dtype=dtype)
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
    return p

# Divided differences for Newton form
def Newton_divided_diff(x, x_values, f, dtype=dtype):
    n = len(x_values)
    y_diff = np.zeros_like(n, dtype=dtype)
    w_prime = np.ones_like(n, dtype=dtype)
    y_values = np.array([f(x) for x in x_values], dtype=dtype)

    for i in range(n+1):
        for j in range(n+1):
            if i != j:
                w_prime[i] *= x_values[i] - x_values[j]
        y_diff[i] += y_values[i] / w_prime[i]

    return y_diff, y_values


# Horner's rule for Newton form
def horners_rule(x, x_values, y_diff):
    n = len(x_values)
    s = y_diff[n-1]
    for i in range(n-1, -1, -1):
        s = s * (x - x_values[i]) + y_diff[i]
    p = s
    return p

def ordering(x_values, flag):
    n = len(x_values)

    # Increasing order
    if flag == 1:
        x_values.sort()

    # Decreasing order
    elif flag == 2:
        x_values.sort(reverse=True)

    # Leja order
    elif flag == 3:
        leja_points = [x_values[0]]
        remaining_points = np.delete(x_values, 0)

        for _ in range(n - 1):
            max_product = -np.inf
            next_point = None
            next_index = -1

            for i, x in enumerate(remaining_points):
                product = np.prod(np.abs(x - np.array(leja_points)))
                if product > max_product:
                    max_product = product
                    next_point = x
                    next_index = i

            leja_points.append(next_point)
            remaining_points = np.delete(remaining_points, next_index)
        x_values = np.copy(leja_points)

    return x_values


# Routine to evaluate error and related statistics
#def evaluate_p()











'SAMPLE POINTS'
# x_values = np.asarray([0, 0.5, 1], dtype=dtype)
# def f5(x): return np.asarray((6*x) + 2, dtype=dtype)
#
# # Compute weights and function values
# gamma, y_values = bary1_weights(x_values, f5)
# print("gamma: ", gamma)
# print("y_values: ", y_values)
#
# # Evaluate interpolation at x = 2
# x_eval = np.asarray(2, dtype=dtype)
# p = bary1_interpolation(x_eval, x_values, gamma, y_values)
#
# print("Interpolated value at x = 2:", p)
