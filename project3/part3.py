import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

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
        G = np.eye(n) - np.diag(1 / D).dot(A) if isinstance(matrix_representation, np.ndarray) else None
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
        if isinstance(matrix_representation, tuple):
            # Construct G for sparse
            G = np.zeros((n, n))  # Not explicitly computed for sparse
        else:
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
                for i in range(n):
                    sigma = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i + 1:], x[i + 1:])
                    x[i] = (b[i] - sigma) / A[i, i]

            r = b - matrix_vector_multiply(x)

            # Increment iteration counter
            iter += 1

    # Gauss-Seidel (Symmetric) Method
    if flag == 3:
        if isinstance(matrix_representation, tuple):
            # Construct G for sparse
            G = np.zeros((n, n))  # Not explicitly computed for sparse
        else:
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
                for i in range(n):
                    sigma = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i + 1:], x[i + 1:])
                    x[i] = (b[i] - sigma) / A[i, i]

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
                for i in range(n - 1, -1, -1):
                    sigma = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i + 1:], x[i + 1:])
                    x[i] = (b[i] - sigma) / A[i, i]

            r = b - matrix_vector_multiply(x)

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
-------------------- Test for Dense and Sparse Matrix --------------------
"""
n = 5
A = sparse_matrix(n)
x_tilde = np.random.randint(1, 10, n)
x0 = np.random.randint(1, 10, n)
b = np.dot(A, x_tilde)  # Dense matrix case

# Dense matrix case
print("Dense Matrix:")
print(A)
x, G, spectral_radius, G_norm, iter, rel_err_arr = stationary_method(A, x_tilde, x0, b, 1)
print("Solution:", x)
print("Iterations:", iter)
print("G:", G)
print("Spectral Radius:", spectral_radius)
print("G_norm:", G_norm)


# Sparse matrix case
AA, JA, IA = compressed_row(A)
print("\nSparse Matrix:")
x, G, spectral_radius, G_norm, iter, rel_err_arr  = stationary_method((AA, JA, IA), x_tilde, x0, b, 1)
print("Solution:", x)
print("Iterations:", iter)
print("G:", G)

