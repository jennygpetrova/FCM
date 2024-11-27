from mypackage import myfunctions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.linalg import cholesky
np.random.seed(123)

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
    n = len(x)
    G = None
    I = np.eye(n)

    rel_err_arr = []
    rel_err = 1

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

            # Count Iteration
            iter += 1

    # Gauss-Seidel (Forward) Method
    if flag == 2:
        # Contraction form
        G = np.linalg.inv(L + np.diag(D)).dot(U)

        while iter < 1000 and rel_err > tol:
            rel_err = np.linalg.norm(x - x_tilde) / np.linalg.norm(x_tilde)
            rel_err_arr.append(rel_err)

            # Forward substitution
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

            # Forward substitution
            x = myfunctions.forward_sweep(A, x, b, n)

            # Backward substitution
            x = myfunctions.backward_sweep(A, x, b, n)

            # Increment iteration counter
            iter += 1

    # Compute eigenvalues, spectral radius, and norm of G
    eigenvalues = np.linalg.eigvals(G)
    spectral_radius = np.max(np.abs(eigenvalues))
    G_norm = np.linalg.norm(G)

    return x, G, spectral_radius, G_norm, iter, rel_err_arr

"""
-------------------- Routine to Generate Test Matrices --------------------
"""
# Choose A from the following predetermined matrices
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

# Run analyses on each matrix A
def analyze_matrix(A, tol=1e-8):
    def is_symmetric(A):
        return np.allclose(A, A.T, atol=tol)

    def is_diagonally_dominant(A):
        D = np.abs(np.diag(A))  # Diagonal elements
        S = np.sum(np.abs(A), axis=1) - D  # Row sums minus diagonal
        return np.all(D >= S)

    def is_spd(A):
        if not is_symmetric(A):
            return False
        try:
            cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False

    def condition_number(A):
        return np.round(np.linalg.cond(A), 2)

    # Perform all checks
    analysis = {
        "Symmetric": is_symmetric(A),
        "Diagonally Dominant": is_diagonally_dominant(A),
        "SPD": is_spd(A),
        "Condition Number": condition_number(A)
    }
    return analysis


"""
-------------------- Function for Generating Results --------------------
"""
def plot_relative_errors(rel_errs, methods, choice):
    plt.figure(figsize=(8, 6))
    for rel_err, method in zip(rel_errs, methods):
        plt.plot(range(len(rel_err)), rel_err, label=method)

    plt.yscale('log')
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Relative Error (log scale)', fontsize=12)
    plt.title(f"Convergence of Relative Errors for Matrix {choice}", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.savefig(f'error_ratios_matrix_{choice}.png', dpi=300, bbox_inches='tight')
    plt.show()


"""
-------------------- Main Routine --------------------
"""
# Initialize a list to store analysis results for all matrices
analysis_results = []

# Loop through matrices
for i in range(9):
    A = matrix_type_dense(i)

    # Perform analysis for the matrix
    analysis = analyze_matrix(A)
    analysis["Matrix"] = i  # Add matrix identifier
    analysis_results.append(analysis)

    # Perform convergence tests and generate plots
    results = []
    rel_errs_all_methods = []
    methods = []

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

        for j in range(5):
            x_tilde = np.random.uniform(-100, 100, A.shape[0])
            x0 = np.random.uniform(-100, 100, A.shape[0])
            b = np.dot(A, x_tilde)
            start_time = time.time()
            x, G, spectral_radius, G_norm, iter, rel_err_arr = stationary_method(A, x_tilde, x0, b, flag)
            end_time = time.time()
            time_list.append(end_time - start_time)
            iter_list.append(iter)
            spectral_radii_list.append(spectral_radius)
            G_norm_list.append(G_norm)

        iter_avg = np.average(iter_list)
        spectral_radius_avg = np.round(np.average(spectral_radii_list), 5)
        G_norm_avg = np.round(np.average(G_norm_list), 5)
        time_avg = np.round(np.average(time_list), 5)

        results.append((method, iter_avg, spectral_radius_avg, G_norm_avg, time_avg))
        rel_errs_all_methods.append(rel_err_arr)
        methods.append(method)

    # Plot relative errors for this matrix
    plot_relative_errors(rel_errs_all_methods, methods, i)

    # Table of convergence results for this matrix
    df = pd.DataFrame(results, columns=['Method', 'Iterations', 'Spectral Radius', 'G Norm', 'Time to Converge'])
    print(f"Results for Matrix {i}:\n")
    print(df)
    df.to_csv(f'Results_Matrix_{i}.csv', index=False)

# Create and display the analysis results table
analysis_df = pd.DataFrame(analysis_results)
analysis_df = analysis_df[["Matrix", "Symmetric", "Diagonally Dominant", "SPD", "Condition Number"]]

analysis_df.to_csv('Matrix_Properties_Analysis.csv', index=False)
print("Matrix Properties Analysis:")
print(analysis_df)
