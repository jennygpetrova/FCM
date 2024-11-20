from mypackage import myfunctions
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1234)

"""
-------------------- Functions for Stationary Methods --------------------
"""
# Jacobi Iteration Method
def jacobi_iteration(A, x_tilde, x0):
    # Diagonal elements of A
    D = np.diag(A)
    if np.any(D == 0):
        raise ValueError("Matrix A contains zero diagonal elements, Jacobi iteration cannot proceed.")

    # Initial values
    x = x0
    resid_arr = []
    err_arr = []

    # Termination criterion
    max_iter = 1000
    tol = 1e-6
    iter = 0

    for _ in range(max_iter):
        b = A * x
        # Compute residual
        r = b - np.dot(A, x)
        resid_arr.append(np.linalg.norm(r))

        err = np.subtract(x_tilde, x)
        err_norm = np.linalg.norm(err)
        err_arr.append(err_norm)

        if err_norm / np.linalg.norm(err) < tol:  # Convergence check
            break

        # Jacobi iteration update
        x += (r / D)

        iter += 1

    return (x, iter, resid_arr)

# Gauss-Seidel (Forward) Iteration Method
def gauss_seidel(A, x_tilde, x0):




def part_2_driver(choice):
    if choice == 0:
        A = np.matrix([[3, 7, -1], [7, 4, 1], [-1, 1, 2]])
    elif choice == 1:
        A = np.matrix([[3, 0, 4], [7, 4, 2], [-1, 1, 2]])
    elif choice == 2:
        A = np.matrix([[-3, 3, -6], [-4, 4, -8], [5, 7, -9]])
    elif choice == 3:
        A = np.matrix([[4, 1, 1], [2, -9, 0], [0, -8, -6]])
    elif choice == 4:
        A = np.matrix([[7,6,9], [4,5,-4], [-7,-3,8]])
    elif choice == 5:
        A = np.matrix([[6,-2,0], [-1,2,-1], [0,-1.2,1]])
    elif choice == 6:
        A = np.matrix([[5,-1,0], [-1,2,-1], [0,-1.5,1]])
    elif choice == 7:
        D = np.array(1,7)


    return A

print(part_2_driver(0))
