import numpy as np
import matplotlib.pyplot as plt
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
                x = np.random.choice([0, 0, 1])
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

def plot_convergence(rel_err_arr):
    plt.plot(rel_err_arr)
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Relative Error (log scale)')
    plt.title('Convergence of Stationary Method')
    plt.grid(True)
    plt.show()

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
        elif flag == 2 or flag == 3:  # Gauss-Seidel or Symmetric Gauss-Seidel
            n = len(x)
            # Forward sweep
            for i in range(n):
                row_start = IA[i] if isinstance(matrix_representation, tuple) else 0
                row_end = IA[i + 1] if isinstance(matrix_representation, tuple) else n
                sigma = sum(AA[k] * x[JA[k]] for k in range(row_start, row_end) if JA[k] != i)
                x[i] = (b[i] - sigma) / D[i]

            if flag == 3:  # Backward sweep for symmetric Gauss-Seidel
                for i in range(len(x) - 1, -1, -1):
                    row_start = IA[i] if isinstance(matrix_representation, tuple) else 0
                    row_end = IA[i + 1] if isinstance(matrix_representation, tuple) else n
                    sigma = sum(AA[k] * x[JA[k]] for k in range(row_start, row_end) if JA[k] != i)
                    x[i] = (b[i] - sigma) / D[i]

        # Update residual
        r = b - matrix_vector_multiply(x)
        iter += 1

    return x, iter, rel_err_arr


"""
-------------------- Test for Dense and Sparse Matrices --------------------
"""
n = 5
A = sparse_matrix(n)
x_tilde = np.random.randint(1, 10, n)
x0 = np.random.randint(1, 10, n)
b = np.dot(A, x_tilde)  # Dense matrix case

# Dense matrix case
print("Dense Matrix:")
print(A)
x, iter, rel_err_arr = stationary_method(A, x_tilde, x0, b, 1)
print("Solution:", x)
print("Iterations:", iter)
plot_convergence(rel_err_arr)

# Sparse matrix case
AA, JA, IA = compressed_row(A)
print("\nSparse Matrix:")
x, iter, rel_err_arr = stationary_method((AA, JA, IA), x_tilde, x0, b, 1)
print("Solution:", x)
print("Iterations:", iter)
plot_convergence(rel_err_arr)

