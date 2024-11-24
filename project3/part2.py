from numpy import ndarray

from mypackage import myfunctions
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(1234)

"""
-------------------- Routine for Generating a Sparse Matrix and Storing in CSR Format --------------------
"""
# Sparse symmetric positive definite (diagonally dominant) matrix
def sparse_matrix(n):
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i > j:
                x = np.random.choice([0, 1])
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
        for col, val in enumerate(row):
            if val != 0:
                AA.append(val)
                JA.append(col)
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
-------------------- Routine for Stationary Iterative Methods --------------------
"""
def stationary_method(matrix_representation, x_tilde, x0, b, flag):
    # Determine whether the matrix is dense or in CSR format
    if isinstance(matrix_representation, tuple):  # CSR format
        AA, JA, IA = matrix_representation
        def matrix_vector_multiply(x):
            return csr_multiply(AA, JA, IA, x)

        def extract_diagonal():
            diag = np.zeros(len(IA) - 1)
            for i in range(len(diag)):
                for k in range(IA[i], IA[i + 1]):
                    if JA[k] == i:
                        diag[i] = AA[k]
                        break
            return diag

        def extract_lower_upper():
            # Extract lower and upper triangular parts of the matrix
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

    else:  # Dense matrix
        A = matrix_representation
        def matrix_vector_multiply(x):
            return np.dot(A, x)

        def extract_diagonal():
            return np.diag(A)

        def extract_lower_upper():
            L = np.tril(A, k=-1)
            U = np.triu(A, k=1)
            return L, U

    def forward_sweep(A, b, x, n):
        for i in range(n - 1, -1, -1):
            sigma = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i + 1:], x[i + 1:])
            x[i] = (b[i] - sigma) / A[i, i]
        return x

    def backward_sweep(A, b, x, n):
        for i in range(n):
            sigma = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i + 1:], x[i + 1:])
            x[i] = (b[i] - sigma) / A[i, i]
        return x

    # Initialize variables
    x = x0.copy()
    r = b - matrix_vector_multiply(x)
    D = extract_diagonal()
    L, U = extract_lower_upper()
    if np.any(D == 0):
        raise ValueError("Matrix A contains zero diagonal elements, iteration cannot proceed.")
    rel_err_arr = []
    rel_err = 1
    n = len(x)

    # Termination criterion
    max_iter = 1000
    tol = 1e-6
    iter = 0

    # Initialize iteration matrix G
    G = None

    # Jacobi Method
    if flag == 1:
        if isinstance(matrix_representation, np.ndarray):
            G = np.eye(n) - np.diag(1 / D).dot(A)

        while iter < max_iter and rel_err > tol:
            rel_err = np.linalg.norm(x - x_tilde) / np.linalg.norm(x_tilde)
            rel_err_arr.append(rel_err)

            # Iteration update
            x_new = x + r / D
            r = b - matrix_vector_multiply(x_new)
            x = x_new

            # Increment iteration counter
            iter += 1

    # Gauss-Seidel (Forward) Method
    if flag == 2:
        if isinstance(matrix_representation, np.ndarray):
            G = np.linalg.inv(L + np.diag(D)).dot(U)

        while iter < max_iter and rel_err > tol:
            rel_err = np.linalg.norm(x - x_tilde) / np.linalg.norm(x_tilde)
            rel_err_arr.append(rel_err)

            if isinstance(matrix_representation, tuple):  # Sparse case
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
            else:  # Dense case
                x = forward_sweep(A, b, x, n)

            # Increment iteration counter
            iter += 1

    # Gauss-Seidel (Symmetric) Method
    if flag == 3:
        if isinstance(matrix_representation, ndarray):
            G_forward = np.linalg.inv(L + np.diag(D)).dot(U)
            G = G_forward.dot(np.linalg.inv(L + np.diag(D)))

        while iter < max_iter and rel_err > tol:
            rel_err = np.linalg.norm(x - x_tilde) / np.linalg.norm(x_tilde)
            rel_err_arr.append(rel_err)

            # Forward sweep
            if isinstance(matrix_representation, tuple):  # Sparse case
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
            else:  # Dense case
                x = forward_sweep(A, b, x, n)

            # Backward sweep
            if isinstance(matrix_representation, tuple):  # Sparse case
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

            else:  # Dense case
                x = backward_sweep(A, b, x, n)

            # Increment iteration counter
            iter += 1

    if isinstance(matrix_representation, tuple):
        spectral_radius = None
        G_norm = None
    else:
        eigenvalues = np.linalg.eigvals(G)
        spectral_radius = np.max(np.abs(eigenvalues))
        G_norm = np.linalg.norm(G)

    return x, G, spectral_radius, G_norm, iter, rel_err_arr


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

flag = 3
if flag == 1:
    method = 'Jacobi'
if flag == 2:
    method = 'Gauss-Seidel'
if flag == 3:
    method = 'Symmetric Gauss-Seidel'

matrix_num = []
iter_list = []
spectral_radii = []
G_norm_list = []
for i in range(9):
    A = part_2_driver(i)
    n = A.shape[0]
    x_tilde = np.random.uniform(-10, 10, n)
    x0 = np.ones(n)
    b = np.dot(A, x_tilde)
    x, G, spectral_radius, G_norm, iter, rel_err_arr = stationary_method(A, x_tilde, x0, b, flag)
    matrix_num.append(i)
    iter_list.append(iter)
    spectral_radii.append(spectral_radius)
    G_norm_list.append(G_norm)

data = list(zip(matrix_num, iter_list, spectral_radii, G_norm_list))
df = pd.DataFrame(data, columns=['Matrix', 'Iterations', 'Spectral Radius', 'G Norm'])
df.to_csv(f'Results for {method} Method.csv', index=False)
print(df)


