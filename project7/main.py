import numpy as np
import math
import matplotlib.pyplot as plt

def composite_newton_cotes(a, b, N, f, num_points, closed=True, alpha=False):
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


def adaptive_trapezoidal(f, a, b, tol=1e-6, maxiter=20):
    # Initial estimate with one subinterval.
    N = 1
    h = b - a  # step size for current partition
    T_old = 0.5 * h * (f(a) + f(b))

    for iter in range(maxiter):
        # Add new points: midpoints of each current subinterval.
        new_sum = 0.0
        for j in range(N):
            x_mid = a + (j + 0.5) * h
            new_sum += f(x_mid)

        # The new composite trapezoidal rule with 2N intervals:
        T_new = 0.5 * T_old + (h / 2) * new_sum

        if abs(T_new - T_old) < tol:
            return T_new

        # Prepare for next iteration.
        T_old = T_new
        N *= 2
        h /= 2

    return T_new


def adaptive_midpoint(f, a, b, tol=1e-6, maxiter=20):
    h = b - a  # current subinterval length (coarse grid)
    N = 1  # number of intervals (initially 1)
    # Initial composite midpoint value:
    S = f(a + h / 2)  # sum over the current grid (one midpoint)
    M_old = h * S  # M_1

    for iter in range(maxiter):
        # In the next refinement, each current interval (of length h)
        # is split into 3 subintervals (new step size h_new = h/3).
        new_sum = 0.0
        for i in range(N):
            x_left = a + i * h
            # Compute the two new midpoints (for the first and third subintervals):
            new_sum += f(x_left + h / 6) + f(x_left + 5 * h / 6)
        # Add the new values to the already computed ones (the previous midpoints)
        # Note: In each old interval the midpoint at x_left+h/2 is already in S.
        S = S + new_sum
        # Now update the subinterval width:
        h_new = h / 3
        # The refined grid now has 3*N subintervals and the composite midpoint
        # estimate is:
        M_new = h_new * S

        if abs(M_new - M_old) < tol:
            return M_new

        # Prepare for the next iteration.
        M_old = M_new
        # Reset the current step size: now each subinterval in the refined grid
        # becomes the "old" block for the next refinement.
        h = h_new
        N *= 3

    return M_new




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

ref1 = adaptive_trapezoidal(f1, a, b)
print(ref1)
ref2 = adaptive_midpoint(f1, a, b)
print(ref2)

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
