import numpy as np
import math
import matplotlib.pyplot as plt

def composite_newton_cotes(a, b, N, f, num_points, closed=True):
    H = (b - a) / N
    sum = 0
    if closed:
        if num_points == 1:  # Left Rectangle Rule
            for i in range(N):
                sum += f(a + (i*H))
            return H * sum
        if num_points == 2:  # Trapezoidal Rule
            for i in range(1, N):
                sum += f(a + (i*H))
            term = f(a) + f(b) + (2 * sum)
            return (H/2) * term
        if num_points == 3:  # Simpson's Rule
            sum2 = 0
            for i in range(1, N):
                if i % 2 == 0:
                    sum += f(a + (i*H))
                else:
                    sum2 += f(a + (i*H))
            term = f(a) + f(b) + (2 * sum) + (4 * sum2)
            return (1/3) * H * term
    else:
        if num_points == 1:  # Midpoint Rule
            k = (1/2)
            for i in range(N):
                sum += f(a + ((i + k) * H))
            return H * sum
        if num_points == 2:  # Two Point Rule
            k = (1/3)
            for i in range(N):
                x1 = a + ((i + k) * H)
                x2 = a + ((i + (2 * k)) * H)
                sum += f(x1) + f(x2)
            return (H/2) * sum

def composite_gauss_legendre(a, b, N, f):
    H = (b - a) / N
    x1 = 1 / np.sqrt(3)
    sum = 0
    for i in range(N):
        a_i = a + (i*H)
        b_i = a + ((i+1)*H)
        term1 = (b_i - a_i) / 2
        term2 = (b_i + a_i) / 2
        sum += f((x1 * term1) + term2) + f((-1 * x1 * term1) + term2)
    return sum * H / 2


"""FUNCTIONS FOR TESTING"""
def f1(x):
    return (math.e ** x)
def f2(x):
    exp = np.sin(2 * x)
    return (math.e ** exp) * np.cos(2 * x)
def f3(x):
    return np.tanh(x)
def f4(x):
    return x * np.cos(2 * np.pi * x)
def f5(x):
    return x + (1/x)


a = 0
b = 3
M = 20



for f in [f1, f2, f3, f4]:
    print("Function", f.__name__)
    result = composite_gauss_legendre(a, b, M, f)
    print("GL:", result)
    for n in range(1,3):
        result1 = composite_newton_cotes(a, b, M, f, num_points=n, closed=False)
        print(f"NC Open n={n}:", result1)
    for n in range(1,4):
        result2 = composite_newton_cotes(a, b, M, f, num_points=n)
        print(f"NC Closed n={n}:", result2)
