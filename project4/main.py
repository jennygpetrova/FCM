import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(123)

'''--------------Functions for Root-Finding Methods--------------'''

def regula_falsi(f, x0, x1, rho, tol=1e-6, max_iter=1000):

    if f(x0) * f(x1) > 0:
        print("Error: The interval does not bracket a root.")
        return None, 0, [], []

    results = []
    errs = []
    x_true = rho

    for i in range(max_iter):
        q = ( f(x1) - f(x0) ) / (x1 - x0)
        if abs(q) < tol:
            return None, i + 1, None, None  # Division by zero, no convergence
        x2 = x1 - (f(x1) / q)

        results.append((i + 1, round(x2,5), round(f(x2),5)))
        err = abs(x2 - x_true)
        errs.append(err)

        if abs(err) < tol:
            return x2, i + 1, results, errs

        if f(x2) * f(x0) < 0:
            x0 = x1
        else:
            x1 = x2

    return x2, max_iter, results, errs


def secant_method(f, x0, x1, rho, tol=1e-6, max_iter=1000):
    results = []
    errs = []
    x_true = rho

    for i in range(max_iter):
        q = ( f(x1) - f(x0) ) / (x1 - x0)

        if abs(q) < tol:
            return None, i + 1, None, None # Division by zero, no convergence
        x2 = x1 - (f(x1) / q)

        results.append((i + 1, round(x2,5), round(f(x2),5)))
        err = abs(x2 - x_true)
        errs.append(err)

        if abs(err) < tol:
            return x2, i + 1, results, errs

        x0, x1 = x1, x2

    return x2, max_iter, results, errs


def newtons_method(f, df, x0, rho, m=1, tol=1e-6, max_iter=1000):
    results = []
    errs = []
    x_true = rho

    for i in range(max_iter):
        x1 = x0 - (m * f(x0) / df(x0))

        results.append((i + 1, round(x1,5), round(f(x1),5)))
        err = abs(x1 - x_true)
        errs.append(err)

        if abs(df(x1)) < tol:
            return x1, i + 1, results, errs # Derivative too small

        if abs(f(x1)) < tol or err < tol:
            return x1, i + 1, results, errs

        x0 = x1

    return x1, max_iter, results, errs


def steffensons_method(f, x0, rho, tol=1e-6, max_iter=1000):
    results = []
    errs = []
    x_true = rho

    for i in range(max_iter):
        denom = ( f( x0 + f(x0) ) ) - f(x0)
        if abs(denom) < tol:  # Check for zero denominator
            print(f"Warning: Denominator too small at iteration {i}, x = {x0}")
            return x1, i + 1, results, errs
        x1 = x0 - f(x0)**2 / denom

        results.append((i + 1, round(x1,5), round(f(x1),5)))
        err = abs(x1 - x_true)
        errs.append(err)

        if err < tol:
            return x1, i + 1, results, errs

        x0 = x1

    return x1, max_iter, results, errs


def table_results(results, method, d, x0):
    df = pd.DataFrame(results, columns=["Iteration", "x_k", "f(x_k)"])
    pd.set_option("display.float_format", "{:.7f}".format)
    print(df)
    x0 = int(x0)
    df.to_csv(f'table-{method}-{d}-{x0}-1.csv', index=False)


def plot_results(results, method, d, x0):
    xk_values = [result[1] for result in results]  # x_k values
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(xk_values)), xk_values, color='g', marker='o', label=f"x_0 = {x0}")
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("x_k (Approximation to Root)", fontsize=12)
    plt.title(f"{method} Method: x_k vs Iterations", fontsize=14)
    plt.legend()
    x0 = int(x0)
    plt.savefig(f'plot-{method}-{d}-{x0}-1.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_errors(errs_method, method, d, x0):
    errs = []
    for i in range(len(errs_method) - 1):
        errs.append(errs_method[i + 1] / (errs_method[i]))

    plt.figure(figsize=(8, 6))
    plt.plot(range(len(errs)), errs, color="r", marker='o', label="Error Ratios")
    plt.plot(range(len(errs_method)), errs_method, color="b", marker='o', label="Error")
    plt.xlabel("Number of iterations", fontsize=12)
    plt.ylabel("Error", fontsize=12)
    plt.legend()
    plt.title(f"Convergence of Errors {method}, d={d}, x0={x0}", fontsize=14)
    x0 = int(x0)
    plt.savefig(f'error-{method}-{d}-{x0}-1.png', dpi=300, bbox_inches='tight')
    plt.show()



'''--------------Testing Root-Finding Methods--------------'''
# Set indicator ('flag') for problems tested
flag = 1

'''Higher Order Roots'''
if flag == 1:
    rho = 2.0
    order = [1, 2, 3, 4, 5]
    initial_intervals = [(-2.5, 2.5), (1.0, 3.0)]
    initial_values = [2.5, 4.0]

    for d in order:
        print("\nd: ", d)

        f = lambda x: (x - rho) ** d
        df = lambda x: d * ((x - rho) ** (d - 1))

        for x0, x1 in initial_intervals:
            print("x0: ", x0)
            # Regula Falsi
            root_rf, iter_rf, results_rf, errs_rf = regula_falsi(f, x0, x1, rho)
            print("Root (Regula Falsi):", root_rf)
            print("Iterations:", iter_rf)
            if errs_rf is not None:
                plot_errors(errs_rf, "Regula Falsi", d, x1)
            table_results(results_rf, "Regula Falsi", d, x1)
            plot_results(results_rf, "Regula Falsi", d, x1)

            # Secant Method
            root_secant, iter_secant, results_secant, errs_secant = secant_method(f, x0, x1, rho)
            print("Root (Secant Method):", root_secant)
            print("Iterations:", iter_secant)
            if errs_secant is not None:
                plot_errors(errs_secant, "Secant", d, x1)
            table_results(results_secant, "Secant", d, x1)


        # for x0 in initial_values:
        #     print("x0: ", x0)
        #
        #     # Newton's Method
        #     root_newton, iter_newton, results_newton, errs_newton = newtons_method(f, df, x0, rho)
        #     print("Root (Newton's Method m=1):", root_newton)
        #     print("Iterations:", iter_newton)
        #     if errs_newton is not None:
        #         plot_errors(errs_newton, "Standard Newton's", d, x0)
        #     table_results(results_newton, "Standard Newton's", d, x0)
        #     plot_results(results_newton, "Standard Newton's", d, x0)
        #
        #     # Modified Newton's Method
        #     root_mod_newton, iter_mod_newton, results_mod_newton, errs_mod_newton = newtons_method(f, df, x0, rho, m=d)
        #     print("Root (Modified Newton's Method m=d):", root_mod_newton)
        #     print("Iterations:", iter_mod_newton)
        #     if errs_mod_newton is not None:
        #         plot_errors(errs_mod_newton, "Modified Newton's (m=d)", d, x0)
        #     table_results(results_mod_newton, "Modified Newton's (m=d)", d, x0)
        #     plot_results(results_mod_newton, "Modified Newton's (m=d)", d, x0)
        #
        #     # Modified Newton's with m > d
        #     m1 = d + 1
        #     root_mod_newton1, iter_mod_newton1, results_mod_newton1, errs_mod_newton1 = newtons_method(f, df, x0, rho, m=m1)
        #     print("Root (Modified Newton's Method m>d):", root_mod_newton1)
        #     print("Iterations:", iter_mod_newton1)
        #     if errs_mod_newton1 is not None:
        #         plot_errors(errs_mod_newton1, "Modified Newton's (m>d)", d, x0)
        #     table_results(results_mod_newton1, "Modified Newton's (m>d)", d, x0)
        #     plot_results(results_mod_newton1, "Modified Newton's (m>d)", d, x0)
        #
        #     # Modified Newton's with m < d
        #     m2 = d - 1
        #     root_mod_newton2, iter_mod_newton2, results_mod_newton2, errs_mod_newton2 = newtons_method(f, df, x0, rho, m=m2)
        #     print("Root (Modified Newton's Method m<d):", root_mod_newton2)
        #     print("Iterations:", iter_mod_newton2)
        #     if errs_mod_newton2 is not None:
        #         plot_errors(errs_mod_newton2, "Modified Newton's (m<d)", d, x0)
        #     table_results(results_mod_newton2, "Modified Newton's (m<d)", d, x0)
        #     plot_results(results_mod_newton2, "Modified Newton's (m<d)", d, x0)
        #
        #
        #     # Steffensen's Method (optional)
        #     root_steff, iter_steff, results_steff, errs_steff = steffensons_method(f, x0, rho)
        #     print("Root (Steffensen's Method):", root_steff)
        #     print("Iterations:", iter_steff)
        #     if errs_steff is not None:
        #         plot_errors(errs_steff, "Steffenson's", d, x0)
        #     table_results(results_steff, "Steffenson's", d, x0)
        #     plot_results(results_steff, "Steffenson's", d, x0)



'''Three Distinct Roots'''
if flag == 2:
    rho = 2.0
    alpha = rho / np.sqrt(3)
    xi = rho / np.sqrt(5)
    initial_intervals = [((rho + alpha) / 2, 3.0), (alpha, 4.0)]
    initial_values = [rho + 1.0, (rho + alpha) / 2, xi - 1e-6]
    '''Scaling'''
    sigma = 10

    f = lambda x: sigma * (x**3 - (x * rho**2))
    df = lambda x: sigma * ((3 * x**2) - rho**2)

    for x0 in initial_values:
        print("x0: ", x0)
        # Newton's Method
        root_newton, iter_newton, results_newton, errs_newton = newtons_method(f, df, x0, rho)
        print("Root (Newton's Method):", root_newton)
        print("Iterations:", iter_newton)
        if errs_newton is not None:
            plot_errors(errs_newton, "Standard Newton's", 1, x0)
        table_results(results_newton, "Standard Newton's", 1, x0)

        # Steffensen's Method (optional)
        root_steff, iter_steff, results_steff, errs_steff = steffensons_method(f, x0, rho)
        print("Root (Steffensen's Method):", root_steff)
        print("Iterations:", iter_steff)
        if errs_steff is not None:
            plot_errors(errs_steff, "Steffenson's", 1, x0)
        table_results(results_steff, "Steffenson's", 1, x0)

    for x0, x1 in initial_intervals:
        print("x0: ", x0)
        # Regula Falsi
        root_rf, iter_rf, results_rf, errs_rf = regula_falsi(f, x0, x1, rho)
        print("Root (Regula Falsi):", root_rf)
        print("Iterations:", iter_rf)
        if errs_rf is not None:
            plot_errors(errs_rf, "Regula Falsi", 1, x1)
        table_results(results_rf, "Regula Falsi", 1, x1)

        # Secant Method
        root_secant, iter_secant, results_secant, errs_secant = secant_method(f, x0, x1, rho)
        print("Root (Secant Method):", root_secant)
        print("Iterations:", iter_secant)
        if errs_secant is not None:
            plot_errors(errs_secant, "Secant", 1, x1)
        table_results(results_secant, "Secant", 1, x1)



'''Root Coalescing'''
if flag == 3:
    rho1_values = [1.0, 0.1, 0.01, 0.001]
    rho2 = 1.5
    rho_values = [(1.0, .5), (.1, .05), (.01, .005)]
    x0 = 2.0
    x1 = 1.0

    for rho1 in rho1_values:
        f = lambda x: x * (x - rho1) * (x - rho2)
        df = lambda x: (3 * x ** 2) - (2 * (rho1 + rho2) * x) + (rho1 * rho2)
        #
        # print("x0: ", x0)
        # # Newton's Method
        # root_newton, iter_newton, results_newton, errs_newton = newtons_method(f, df, x0, rho2)
        # print("Root (Newton's Method):", root_newton)
        # print("Iterations:", iter_newton)
        # method = f"Newton's (rho1={rho1}, rho2={rho2})"
        # if errs_newton is not None:
        #     plot_errors(errs_newton, method, 1, x0)
        # table_results(results_newton, method, 1, x0)

        # Secant Method
        root_secant, iter_secant, results_secant, errs_secant = secant_method(f, x0, x1, rho2)
        print("Root (Secant Method):", root_secant)
        print("Iterations:", iter_secant)
        method = f"Secant (rho={rho1})"
        if errs_secant is not None:
            plot_errors(errs_secant, method, 1, x0)
        table_results(results_secant, method, 1, x0)
        plot_results(results_secant, method, 1, x0)
