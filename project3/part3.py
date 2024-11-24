import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from mypackage import myfunctions

"""
-------------------- Routine for Generating a Sparse Matrix and Storing in CSR Format --------------------
"""
# Sparse symmetric positive definite (diagonally dominant) matrix
def sparse_matrix(n):
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i > j:
                x = np.random.choice([0, 0, 0, 1])
                if x == 1:
                    # Scale each row to ensure diagonal dominance
                    L[i][j] = np.round(np.random.randint(1, 10) / i, 3)
    A = L + L.T
    np.fill_diagonal(A, 20)
    return A

# Compressed Sparse Row Storage (CSR)
def compressed_row(A):
    AA = []  # Non-zero values
    JA = []  # Column indices
    IA = [0]  # Row pointers
    for row in A:
        for j, a in enumerate(row):
            if a != 0:
                AA.append(a)
                JA.append(j)
        IA.append(len(AA))  # End of current row in AA
    return np.array(AA), np.array(JA), np.array(IA)

# Matrix-vector multiplication for CSR matrices
def csr_multiply(AA, JA, IA, x):
    b = np.zeros(len(IA) - 1)  # Result vector
    for i in range(len(b)):
        for k in range(IA[i], IA[i + 1]):
            b[i] += AA[k] * x[JA[k]]
    return b

"""
-------------------- Generalized Routines for Iterative Methods --------------------
"""
def get_diagonal_dense(A):
    return np.diag(A)

def get_diagonal_sparse(AA, JA, IA):
    diag = np.zeros(len(IA) - 1)
    for i in range(len(diag)):
        for k in range(IA[i], IA[i + 1]):
            if JA[k] == i:
                diag[i] = AA[k]
                break
    return diag

def get_lower_upper_dense(A):
    L = np.tril(A, k=-1)
    U = np.triu(A, k=1)
    return L, U

def get_lower_upper_sparse(AA, JA, IA):
    L = np.zeros(len(IA) - 1)
    U = np.zeros(len(IA) - 1)
    for i in range(len(IA) - 1):
        for k in range(IA[i], IA[i + 1]):
            j = JA[k]
            if j < i:
                L[i] += AA[k]
            elif j > i:
                U[i] += AA[k]
    return L, U

"""def plot_convergence(rel_err_arr):
    plt.plot(rel_err_arr)
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Relative Error (log scale)')
    plt.title('Convergence of Stationary Method')
    plt.grid(True)
    plt.show()"""

def stationary_method(matrix_representation, x_tilde, x0, b, flag):
    if isinstance(matrix_representation, tuple):  # CSR format
        AA, JA, IA = matrix_representation
        matrix_vector_multiply = lambda x: csr_multiply(AA, JA, IA, x)
        get_diagonal = lambda: get_diagonal_sparse(AA, JA, IA)
    else:  # Dense matrix
        A = matrix_representation
        matrix_vector_multiply = lambda x: np.dot(A, x)
        get_diagonal = lambda: get_diagonal_dense(A)

    # Initialize variables
    x = x0.astype(float)
    n = len(x)
    r = b - matrix_vector_multiply(x)
    D = get_diagonal()
    if np.any(D == 0):
        raise ValueError("Matrix A contains zero diagonal elements, iteration cannot proceed.")
    rel_err_arr = []
    rel_err = 1
    max_iter = 1000
    tol = 1e-6
    iter = 0

    # Iterative methods
    while iter < max_iter and rel_err > tol:
        rel_err = np.linalg.norm(x - x_tilde) / np.linalg.norm(x_tilde)
        rel_err_arr.append(rel_err)

        if flag == 1:  # Jacobi
            x += r / D
            r = b - matrix_vector_multiply(x)

        elif flag == 2 or flag == 3:  # Gauss-Seidel or Symmetric Gauss-Seidel
            # Forward sweep
            if isinstance(matrix_representation, tuple):
                for i in range(n):
                    row_start = IA[i]
                    row_end = IA[i + 1]
                    sigma = 0
                    for k in range(row_start, row_end):
                        j = JA[k]
                        if j < i:
                            sigma += AA[k] * x[j]
                        elif j > i:
                            sigma += AA[k] * x[j]
                    x[i] = (b[i] - sigma) / D[i]
            else:
                x = myfunctions.forward_sweep(A, x, b, n)

        if flag == 3:  # Backward sweep for symmetric Gauss-Seidel
            if isinstance(matrix_representation, tuple):
                for i in range(n - 1, -1, -1):
                    row_start = IA[i]
                    row_end = IA[i + 1]
                    sigma = 0
                    for k in range(row_start, row_end):
                        j = JA[k]
                        if j < i:
                            sigma += AA[k] * x[j]
                        elif j > i:
                            sigma += AA[k] * x[j]
                    x[i] = (b[i] - sigma) / D[i]
            else:
                x = myfunctions.backward_sweep(A, x, b, n)


        iter += 1

    return x, iter, rel_err_arr

def plot_relative_errors(rel_errs, methods, n, type):
    plt.figure(figsize=(8, 6))
    for rel_err, method in zip(rel_errs, methods):
        plt.plot(range(len(rel_err)), rel_err, label=method)

    plt.yscale('log')
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Relative Error (log scale)', fontsize=12)
    plt.title(f"Convergence of Relative Errors for {type} Matrix (n = {n})", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.savefig(f'rel_errs_{n}.png', dpi=300, bbox_inches='tight')
    plt.show()


"""
-------------------- Input Collection --------------------
"""
def get_user_inputs():
    print("\nEnter range of dimensions to generate (nxn) matrix A: ")
    nmin = int(input("Minimum value: "))
    nmax = int(input("Maximum value: "))
    step = int(input("Step size: "))

    print("\nEnter range to generate random values for solution vector and initial guess vector:")
    xmin = float(input("Minimum value: "))
    xmax = float(input("Maximum value: "))


    return nmin, nmax, step, xmin, xmax

"""
-------------------- Test for Dense and Sparse Matrices --------------------
"""
nmin, nmax, step, xmin, xmax = get_user_inputs()

for n in range(nmin, nmax + 1, step):
    results = []  # To store results for all methods for this matrix
    results_csr = []
    rel_errs = []  # To store relative errors for all methods
    rel_errs_csr = []
    methods = []  # To store method names

    # Sparse matrix generation
    A = sparse_matrix(n)
    # Compressed storage
    AA, JA, IA = compressed_row(A)

    # Loop through methods
    for flag in range(1, 4):
        if flag == 1:
            method = 'Jacobi'
        elif flag == 2:
            method = 'GS'
        elif flag == 3:
            method = 'SGS'

        iter_list = []
        iter_avg = []
        time_list = []
        time_avg = []
        time_list_csr = []
        time_csr_avg = []

        # Run the method multiple times to average results
        for j in range(5):
            x_tilde = np.random.uniform(-10, 10, n)
            x0 = np.random.uniform(-10, 10, n)

            # Sparse matrix multiplication
            b = np.dot(A, x_tilde)
            # Time for sparse matrix to converge
            start_time = time.time()
            x, iter_num, rel_err_arr = stationary_method(A, x_tilde, x0, b, flag)
            end_time = time.time()
            time_list.append(end_time - start_time)
            iter_list.append(iter_num)

            # Compressed matrix multiplication
            b_csr = csr_multiply(AA, JA, IA, x_tilde)
            start_time = time.time()
            x_csr, iter_csr, rel_err_csr = stationary_method((AA, JA, IA), x_tilde, x0, b_csr, flag)
            end_time = time.time()
            time_list_csr.append(end_time - start_time)

        # Compute averages
        iter_avg = np.average(iter_list)
        time_avg = np.average(time_list)
        time_csr_avg = np.average(time_list_csr)

        # Append results for this method to the list
        results.append((method, iter_avg, time_avg, time_csr_avg))

        # Collect relative errors and method name for plotting
        rel_errs.append(rel_err_arr)
        rel_errs_csr.append(rel_err_csr)
        methods.append(method)

    # Plot relative errors for this method
    plot_relative_errors(rel_errs, methods, n, 'Sparse')
    plot_relative_errors(rel_errs_csr, methods, n, 'Compressed')

    # Create a DataFrame for this matrix
    df = pd.DataFrame(results, columns=['Method', 'Iterations', 'Sparse Time', 'Compressed Time'])

    # Print and save the results for this matrix
    print(f"Comparing Convergence Times for Sparse vs Compressed Row Storage(n = {n}):\n")
    print(df)
    print("\n")
    df.to_csv(f'Results_Matrix_{n}.csv', index=False)




