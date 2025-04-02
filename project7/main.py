import numpy as np
import math

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

def gauss_legendre(a, b, f):
    H1 = (b - a) / 2
    H2 = (b + a) / 2
    x1 = - 1 / np.sqrt(3)
    x2 = 1 / np.sqrt(3)
    sum = f((H1 * x1) + H2) + f((H1 * x2) + H2)
    return H1 * sum



a = 0.1
b = 1.3
M = 4
def f(x):
    return 5 * x * (math.e ** (- 2 * x))

result = composite_newton_cotes(a, b, M, f, num_points=2, closed=False)
print(result)
result2 = gauss_legendre(a, b, f)
print(result2)



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







