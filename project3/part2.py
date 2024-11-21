from mypackage import myfunctions
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1234)

"""
-------------------- Functions for Stationary Iterative Methods --------------------
"""

# Jacobi Iteration Method
def jacobi_iteration(A, x_tilde, x0, b):
    # Diagonal elements of A
    D = np.diagonal(A)
    if np.any(D == 0):
        raise ValueError("Matrix A contains zero diagonal elements, Jacobi iteration cannot proceed.")

    # Preconditioner
    P = D

    # Initial residual
    r = b - np.dot(A, x0)
    r0_norm = np.linalg.norm(r)
    resid_arr = [r0_norm]
    r_norm = r0_norm

    # Initial error
    err = x0 - x_tilde
    err_A_norm =  np.sqrt(np.sum(A * (err ** 2)))
    err_arr = [err_A_norm]
    err_ratio = [1.0]

    # Initial values
    x = x0.copy()

    # Termination criterion
    max_iter = 1000
    tol = 1e-6
    iter = 0

    while iter < max_iter and r_norm / r0_norm > tol:
        # Iteration update
        x += r / D

        # Compute residual
        r = b - np.dot(A, x)

        # Store residual norms
        r_norm = np.linalg.norm(r)
        resid_arr.append(r_norm)

        # Store error
        err = x - x_tilde
        err_A_norm_next = np.sqrt(np.sum(A * (err ** 2)))
        err_arr.append(err_A_norm_next)
        err_ratio.append(err_A_norm_next / err_A_norm)

        # Keep error term at current step after calculating error ratio
        err_A_norm = err_A_norm_next

        # Increment iteration counter
        iter += 1

    return x, iter, resid_arr, err_arr, err_ratio

# Gauss-Seidel (Forward) Iteration Method
def gauss_seidel(A, x_tilde, x0, b):
    # Diagonal elements of A
    D = np.diag(A)
    # Lower triangular elements of A
    L = np.tril(A)


    # Preconditioner
    P = D - L

    # Initial Values
    x = x0
    resid_arr = []
    err_arr = []
    n = x.size

    # Termination criterion
    max_iter = 1000
    tol = 1e-6
    iter = 0

    for _ in range(max_iter):
        # Compute residual
        r = b - np.dot(A, x)
        resid_arr.append(np.linalg.norm(r))

        err = np.subtract(x_tilde, x)
        err_norm = np.linalg.norm(err)
        err_arr.append(err_norm)

        if err_norm / np.linalg.norm(err) < tol:  # Convergence check
            break

        # Jacobi iteration update
        delta_x = myfunctions.lower_solve(P, r, n)
        x += delta_x

        iter += 1

    return x, iter, resid_arr, err_arr


"""
-------------------- Routine to Generate Test Matrices --------------------
"""
def part_2_driver(choice):
    if choice == 0:
        A = np.array([
            [3, 7, -1],
            [7, 4, 1],
            [-1, 1, 2]
        ])
    if choice == 1:
        A = np.array([
            [3, 0, 4],
            [7, 4, 2],
            [-1, -1, 2]
        ])
    if choice == 2:
        A = np.array([
            [-3, 3, -6],
            [-4, 7, -8],
            [5, 7, -9]
        ])
    if choice == 3:
        A = np.array([
            [4, 1, 1],
            [2, -9, 0],
            [0, -8, -6]
        ])
    if choice == 4:
        A = np.array([
            [7, 6, 9],
            [4, 5, -4],
            [-7, -3, 8]
        ])
    if choice == 5:
        A = np.array([
            [6, -2, 0],
            [-1, 2, -1],
            [0, -6/5, 1]
        ])
    if choice == 6:
        A = np.array([
            [5, -1, 0],
            [-1, 2, -1],
            [0, -3/2, 1]
        ])
    if choice == 7:
        A = np.array([
            [4, -1, 0, 0, 0, 0, 0],
            [-1, 4, -1, 0, 0, 0, 0],
            [0, -1, 4, -1, 0, 0, 0],
            [0, 0, -1, 4, -1, 0, 0],
            [0, 0, 0, -1, 4, -1, 0],
            [0, 0, 0, 0, -1, 4, -1],
            [0, 0, 0, 0, 0, -1, 4]
        ])
    if choice == 8:
        A = np.array([
            [2, -1, 0, 0, 0, 0, 0],
            [-1, 2, -1, 0, 0, 0, 0],
            [0, -1, 2, -1, 0, 0, 0],
            [0, 0, -1, 2, -1, 0, 0],
            [0, 0, 0, -1, 2, -1, 0],
            [0, 0, 0, 0, -1, 2, -1],
            [0, 0, 0, 0, 0, -1, 2]
        ])

    return A


A= part_2_driver(0)
print(A)
x_tilde = np.random.uniform(-10, 10, 3)
x0 = np.ones(3)
b = np.dot(A, x_tilde)
x, iter, resid_arr, err_arr, err_ratio = jacobi_iteration(A, x_tilde, x0, b)
print(x)
print(iter)
