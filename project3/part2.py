from mypackage import myfunctions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
np.random.seed(1234)

"""
-------------------- Routine for Stationary Iterative Methods --------------------
"""

def stationary_method(A, x_tilde, x0, b, flag):
    # Initialize variables
    x = x0
    r = b - np.dot(A, x)
    D = np.diag(A)
    L = np.tril(A, k=-1)
    U = np.triu(A, k=1)
    if np.any(D == 0):
        raise ValueError("Matrix A contains zero diagonal elements, Jacobi iteration cannot proceed.")
    rel_err_arr = []
    rel_err = 1
    n = len(x)
    G = None
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
        G = np.linalg.inv(L + np.diag(D)).dot(U)

        while iter < 1000 and rel_err > tol:
            rel_err = np.linalg.norm(x - x_tilde) / np.linalg.norm(x_tilde)
            rel_err_arr.append(rel_err)

            # Iteration update
            x = myfunctions.forward_sweep(A, x, b, n)

            # Increment iteration counter
            iter += 1

    # Gauss-Seidel (Symmetric) Method
    if flag == 3:
        # Contraction form
        G_forward = np.linalg.inv(L + np.diag(D)).dot(U)
        G = G_forward.dot(np.linalg.inv(L + np.diag(D)))

        while iter < 1000 and rel_err > tol:
            rel_err = np.linalg.norm(x - x_tilde) / np.linalg.norm(x_tilde)
            rel_err_arr.append(rel_err)

            # Iteration update
            x = myfunctions.forward_sweep(A, x, b, n)
            x = myfunctions.backward_sweep(A, x, b, n)

            # Increment iteration counter
            iter += 1

    eigenvalues = np.linalg.eigvals(G)
    spectral_radius = np.max(np.abs(eigenvalues))
    G_norm = np.linalg.norm(G)

    return x, G, spectral_radius, G_norm, iter, rel_err_arr

"""
-------------------- Routine to Generate Test Matrices --------------------
"""
def matrix_type_dense(choice):
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


def plot_relative_errors(rel_err, method, choice):
    plt.figure(figsize=(8, 6))  # Create a new figure
    plt.plot(range(len(rel_err)), rel_err, label=method)

    # Add labels, title, and legend
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel("Relative Error", fontsize=12)
    plt.yscale('log')
    plt.title(f"{method} Convergence of Relative Errors for Matrix {choice}", fontsize=14)
    plt.legend()
    plt.grid(True)

    # Save and display the plot
    plt.savefig(f'error_ratios_matrix_{i}.png', dpi=300, bbox_inches='tight')
    plt.show()


"""
-------------------- Main Routine --------------------
"""
# Loop through matrices
for i in range(9):
    matrix_num = i
    results = []  # To store results for all methods for this matrix

    A = matrix_type_dense(i)
    n = A.shape[0]

    # Loop through methods
    for flag in range(1, 4):
        if flag == 1:
            method = 'Jacobi'
        elif flag == 2:
            method = 'GS'
        elif flag == 3:
            method = 'SGS'

        iter_list = []
        spectral_radii_list = []
        G_norm_list = []
        time_list = []

        # Run the method multiple times to average results
        for j in range(5):
            x_tilde = np.random.uniform(-10, 10, n)
            x0 = np.random.uniform(-10, 10, n)
            b = np.dot(A, x_tilde)
            start_time = time.time()
            x, G, spectral_radius, G_norm, iter, rel_err_arr = stationary_method(A, x_tilde, x0, b, flag)
            end_time = time.time()
            time_list.append(end_time - start_time)
            iter_list.append(iter)
            spectral_radii_list.append(spectral_radius)
            G_norm_list.append(G_norm)

        # Compute averages for this method
        iter_avg = np.average(iter_list)
        spectral_radius_avg = np.average(spectral_radii_list)
        G_norm_avg = np.average(G_norm_list)
        time_avg = np.average(time_list)

        # Append results for this method to the list
        results.append((method, iter_avg, spectral_radius_avg, G_norm_avg, time_avg))

        # Plot relative errors for this method
        plot_relative_errors(rel_err_arr, method, i)

    # Create a DataFrame for this matrix
    df = pd.DataFrame(results, columns=['Method', 'Iterations', 'Spectral Radius', 'G Norm', 'Time to Converge'])

    # Print and save the results for this matrix
    print(f"Results for Matrix {i}:\n")
    print(df)
    print("\n")
    df.to_csv(f'Results_Matrix_{i}.csv', index=False)
