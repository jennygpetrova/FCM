import numpy as np
from numpy import dtype

# Preset datatype to change between 32-bit and 64-bit floating point system
dtype=np.float64
#dtype = np.float32

# Functions f(x) for testing routines
def f(x, dtype=dtype):
    x = np.asarray(x, dtype=dtype)
    y = (x**2) + (2*x) + 9
    return y

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
    l = np.ones_like(x, dtype=dtype)
    for i in range(n):
        term = np.ones_like(x, dtype=dtype)
        for j in range(n):
            if i != j:
                term *= (x - x_values[j]) / (x_values[i] - x_values[j])
        l += term
    return l

def f4(x, dtype=dtype):
    x = np.asarray(x, dtype=dtype)
    y = 1 / (1 + (25 * x ** 2))
    return y

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

def bary1_interpolation(x, x_values, gamma, y_values, dtype=dtype):
    x = np.asarray(x, dtype=dtype)
    x_values = np.asarray(x_values, dtype=dtype)
    numerator = np.zeros_like(x, dtype=dtype)
    denominator = np.zeros_like(x, dtype=dtype)

    for i in range(len(x_values)):
        if np.abs(x - x_values[i]) < np.finfo(dtype).eps:
            return y_values[i]
        term = gamma[i] / (x - x_values[i])
        numerator += term * y_values[i]
        denominator += term

    return numerator / denominator

# Sample points
x_values = np.asarray([0, 0.5, 1], dtype=dtype)

def f5(x): return np.asarray(x, dtype=dtype)

# Compute weights and function values
gamma, y_values = bary1_weights(x_values, f5)

# Evaluate interpolation at x = 2
x_eval = np.asarray(2, dtype=dtype)
p = bary1_interpolation(x_eval, x_values, gamma, y_values)

print("Interpolated value at x = 2:", p)
