from mypackage import myfunctions
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1234)

"""
-------------------- Routine for Stationary Iterative Methods --------------------
"""

def stationary_method(A, x_tilde, x0, b, flag):
    # Initialize variables
    x = x0
    r = b - np.dot(A, x)
    D = np.diag(A)
    if np.any(D == 0):
        raise ValueError("Matrix A contains zero diagonal elements, Jacobi iteration cannot proceed.")
    L = np.tril(A, k=-1)
    U = np.triu(A, k=1)
    rel_err_arr = []
    rel_err = 1
    n = len(x)
    I = np.eye(n)

    # Termination criterion
    max_iter = 1000
    tol = 1e-6
    iter = 0

    # Jacobi Method
    if flag == 1:
        # Preconditioner
        P = D
        # Contraction form
        G = I - (A/D)
        while iter < max_iter and rel_err > tol:
            rel_err = np.linalg.norm(x - x_tilde) / np.linalg.norm(x_tilde)
            rel_err_arr.append(rel_err)

            # Iteration update
            x += r / P

            # Compute residual
            r = b - np.dot(A, x)

            # Increment iteration counter
            iter += 1

    # Gauss-Seidel (Forward) Method
    if flag == 2:
        # Contraction form
        G = myfunctions.forward_sweep_GS(A, I, A, n)
        while iter < 1000 and rel_err > tol:
            rel_err = np.linalg.norm(x - x_tilde) / np.linalg.norm(x_tilde)
            rel_err_arr.append(rel_err)

            # Iteration update
            x = myfunctions.forward_sweep_GS(A, x, b, n)

            # Increment iteration counter
            iter += 1

    # Gauss-Seidel (Symmetric) Method
    if flag == 3:
        # Contraction form
        G = myfunctions.forward_sweep_GS(A, I, A, n)
        G = myfunctions.backward_sweep_GS(A, I, b, n)
        while iter < 1000 and rel_err > tol:
            rel_err = np.linalg.norm(x - x_tilde) / np.linalg.norm(x_tilde)
            rel_err_arr.append(rel_err)

            # Iteration update
            x = myfunctions.forward_sweep_GS(A, x, b, n)
            x = myfunctions.backward_sweep_GS(A, x, b, n)

            # Increment iteration counter
            iter += 1

    return x, G, iter, rel_err_arr


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

A = part_2_driver(7)
print(A)
x_tilde = np.random.uniform(-10, 10, 7)
print("x_tilde", x_tilde)
x0 = np.ones(7)
b = np.dot(A, x_tilde)
x, G, iter_num, rel_err_list = stationary_method(A, x_tilde, x0, b, 1)
G_norm = np.linalg.norm(G)
G_eigen = np.linalg.eigvals(G)
G_spectral = np.max(np.abs(G_eigen))
print(x)
print(iter_num)
print(G_norm)
print(G_spectral)
