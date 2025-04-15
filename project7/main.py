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

def adaptive_midpoint(a, b, N, f, true_est, tol=1e-6, maxiter=20):
    H = (b - a) / N
    sum = 0
    k = (1/2)
    for i in range(N):
        sum += f(a + (i + k) * H)
    I = H * sum
    for iteration in range(maxiter):
        H_new = H / 3
        N_new = N * 3
        sum_new = 0
        for i in range(N):
            base = a + i * (3 * H_new)
            x_left = base + k * H_new
            x_right = base + (k+2) * H_new
            sum_new += f(x_left) + f(x_right)
        sum += sum_new
        I_new = H_new * sum
        err = (9 / 8) * (I_new - I)
        est = true_est - I
        print(f"{N} & {I:.8f} & {err:.8f} & {est:.8f}", r"\\")
        if abs(err) < tol:
            return I_new, N_new
        I = I_new
        N = N_new
        H = H_new
    return I, N_new


def adaptive_trapezoidal(a, b, N, f, true_est, tol=1e-6, maxiter=20):
    H = (b - a) / N
    sum = ( f(a) + f(b) ) / 2
    for i in range(1, N):
        sum += f(a + (i * H))
    I = H * sum
    for iteration in range(maxiter):
        N_new = N * 2
        H_new = H / 2
        for i in range(1, N_new, 2):
            sum += f(a + (i * H_new))
        I_new = H_new * sum
        err = (4/3) * (I_new - I)
        est = true_est - I
        print(f"{N} & {I:.8f} & {err:.8f} & {est:.8f}", r"\\")
        if abs(err) < tol:
            return I_new, N_new
        I = I_new
        N = N_new
        H = H_new
    return I, N_new

"""FUNCTIONS FOR TESTING"""
def f1(x):
    return (math.e ** x)
def f2(x):
    exp = np.sin(2 * x)
    return (math.e ** exp) * np.cos(2 * x)
def f3(x):
    return math.tanh(x)
def f4(x):
    return x * np.cos(2 * np.pi * x)
def f5(x):
    return x + (1/x)
def f6(x):
    return 2 ** x
def f7(x):
    return (x ** 3) - (6 * (x ** 2)) + (12 * x) - 8
def f8(x):
    return 2 * x

"CORRESPONDING TRUE VALUES"
f1_true = math.e ** 3 - 1
# print(f1_true)
f2_true = 0.5 * (-1 + math.e ** (np.sqrt(3)/2))
# print(f2_true)
f3_true = math.log(math.cosh(1)/math.cosh(2))
# print(f3_true)
f4_true = - 1 / (2 * np.pi ** 2)
# print(f4_true)
f5_true = (2.5 ** 2 - 0.1 ** 2) / 2 + math.log(2.5/0.1)
# print(f5_true)
f6_true = 15 / math.log(2)
#print(f6_true)
f7_true = 0
# print(f7_true)
f8_true = 0

"TABLE OUTPUTS"
a = -1
b = 1
M_list = [5, 10, 20, 40, 80]
N = 1
results_open = []
results_closed = []
for M in M_list:
    # Open methods
    nc_open1 = composite_newton_cotes(a, b, M, f8, num_points=1, closed=False)
    err1 = f8_true - nc_open1
    nc_open2 = composite_newton_cotes(a, b, M, f8, num_points=2, closed=False)
    err2 = f8_true - nc_open2
    gl = composite_gauss_legendre(a, b, M, f8)
    err3 = f8_true - gl
    results_open.append((M, nc_open1, err1, nc_open2, err2, gl, err3))
    # Closed methods
    nc_closed1 = composite_newton_cotes(a, b, M, f8, num_points=1, closed=True)
    err4 = f8_true - nc_closed1
    nc_closed2 = composite_newton_cotes(a, b, M, f8, num_points=2, closed=True)
    err5 = f8_true - nc_closed2
    nc_closed3 = composite_newton_cotes(a, b, M, f8, num_points=3, closed=True)
    err6 = f8_true - nc_closed3
    results_closed.append((M, nc_closed1, err4, nc_closed2, err5, nc_closed3, err6))


# Table for Open Methods
latex_table_open = r"""\begin{table}[H]
\begin{adjustwidth}{-2.5cm}{}
\caption{Approximations using Open Interval Methods}
\begin{tabular}{c c c c c c c}
\hline
$N$ & Midpoint ($n=1$) & $E(f)$ & 2-Point ($n=2$) & $E(f)$ & Gauss-Legendre ($n=2$) & $E(f)$ \\
\hline
"""
for row in results_open:
    M, nc1, err1, nc2, err2, gl2, err3 = row
    latex_table_open += f"{M}  & {nc1:.8f} & {err1:.8f} & {nc2:.8f} & {err2:.8f} & {gl2:.8f} & {err3:.8f} \\\\\n"
latex_table_open += r"\hline" + "\n" + r"\end{tabular}" + "\n" + r"\end{adjustwidth}" + "\n" + r"\end{table}"

print(latex_table_open)


# Table for Closed Methods
latex_table_closed = r"""\begin{table}[H]
\begin{adjustwidth}{-2.5cm}{}
\caption{Approximations using Closed Interval Methods}
\begin{tabular}{c c c c c c c}
\hline
$N$ & Left Rectangle ($n=1$) & $E(f)$ & Trapezoidal ($n=2$) & $E(f)$ & Simpson's ($n=3$) & $E(f)$ \\
\hline
"""
for row in results_closed:
    M, nc1, err1, nc2, err2, nc3, err3 = row
    latex_table_closed += f"{M}  & {nc1:.8f} & {err1:.8f} & {nc2:.8f} & {err2:.8f} & {nc3:.8f} & {err3:.8f} \\\\\n"
latex_table_closed += r"\hline" + "\n" + r"\end{tabular}" + "\n" + r"\end{adjustwidth}" + "\n" + r"\end{table}"

print(latex_table_closed)


# Table for Adaptive Methods
print(f"N & $I_n$  & $I - I_n$ & Error Estimate \hline")
mdpt, N_mdpt = adaptive_midpoint(a, b, N, f8, f8_true, tol=1e-6, maxiter=20)
print(f"\hline")
print(f"N & $I_n$  & $I - I_n$ & Error Estimate \hline")
trap, N_trap = adaptive_trapezoidal(a, b, N, f8, f8_true, tol=1e-6, maxiter=20)
print(f"\hline")
