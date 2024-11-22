import warnings
import numpy as np
from numpy import linalg as LA

def LU_decomp(A, n, flag):
    Pr = np.arange(n)  # 1D-array to store row permutations (partial pivoting, complete pivoting)
    Pc = np.arange(n)  # 1D-array to store column permutations (complete pivoting)

    for i in range(n):
        if flag == 1:  # No pivoting
            if A[i][i] == 0:  # Pivot element too small
                warnings.warn(f"({i},{i}) element too small", UserWarning)
                return None, None, None  # Factorization will not proceed

        elif flag == 2:  # Partial pivoting (row pivoting)
            if np.all(A[:, i]) == 0:  # Elements in column i too small
                warnings.warn(f"(column {i} has all elements too small", UserWarning)
                return None, None, None  # Factorization should not proceed
            # Find row with element of max magnitude in column i
            max_row = np.argmax(np.abs(A[i:, i])) + i
            if i != max_row:
                A[[i, max_row], :] = A[[max_row, i], :]  # Swap rows in A
                Pr[[i, max_row]] = Pr[[max_row, i]]  # Track row swaps in permutation array

        elif flag == 3:  # Complete pivoting (row and column pivoting)
            # Find position of the element with max magnitude in submatrix A[i:, i:]
            max_row, max_col = divmod(np.argmax(np.abs(A[i:, i:])), n - i)
            max_row += i
            max_col += i

            if A[max_row, max_col] == 0:  # All elements in current submatrix too small
                warnings.warn(f"({max_row},{max_col}) element too small", UserWarning)
                return None, None, None  # Factorization should not proceed

            if max_row != i:
                A[[i, max_row], :] = A[[max_row, i], :] # Swap rows in A
                Pr[[i, max_row]] = Pr[[max_row, i]]  # Track row swaps in permutation array

            if max_col != i:
                A[:, [i, max_col]] = A[:, [max_col, i]] # Swap columns in A
                Pc[[i, max_col]] = Pc[[max_col, i]]  # Track column swaps in permutation array

        # Perform LU factorization on current column i
        for j in range(i+1, n):
            lam = A[j][i] / A[i][i]  # Compute lambda
            A[j][i] = lam  # Store lambda below the diagonal for L
            for k in range(i+1, n):
                A[j][k] -= lam * A[i][k]  # Update entries above the diagonal for U

    # Return the permutation arrays and the modified matrix
    return Pr, Pc, A

# In-place L*U matrix multiplication routine (from Assignment 1)
def evaluate_LU(A, n):
    M = np.zeros((n, n))
    M[0, :n] = A[0, :n]
    M[1:n, :n] = np.outer(A[1:n, 0], A[0, :n])
    if n > 1:
        for i in range(1, n):
            M[i, i:n] += A[i, i:n]
            M[i + 1:n, i:n] += np.outer(A[i + 1:n, i], A[i, i:n])

    return M

# Compute A <- PrAPc (apply permutation arrays to A)
def compute_PrAPc(A_copy, Pr, Pc):
    A_row = A_copy[Pr, :]  # Permutes the rows according to Pr (unchanged if no pivoting)
    A_col = A_row[:, Pc]  # Permutes the columns according to Pc (unchanged if no or partial pivoting)
    A_perm = np.copy(A_col)
    return A_perm

# Solve Ax=b for x using forward and backward substitution
def lower_solve(A, b, n):
    # Forward substitution to solve Ly = b for y (where L is stored in A)
    y = np.zeros(n)
    y[0] = b[0]
    #for i in range(1, n):
    #    y[i] = b[i] - np.dot(A[i, :i], y[:i])

    for i in range(n):
        if A[i, i] == 0:
            raise ValueError(f"Zero diagonal element encountered at index {i}, cannot solve.")
        # Compute the ith element of x
        y[i] = (b[i] - np.dot(A[i, :i], y[:i])) / A[i, i]

    return y

def upper_solve(A, y, n):
    # Backward substitution to solve Ux = y for x (where U is stored in A)
    x = np.zeros(n)
    x[n - 1] = y[n - 1] / A[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (y[i] - np.dot(A[i, i + 1:n], x[i + 1:n])) / A[i, i]

    return x

def forward_sweep_GS(A, x, b, n):
    for i in range(n):
        sum = 0
        for j in range(n):
            if j != i:
                sum += A[i, j] * x[j]
        x[i] =  (b[i] - sum) / A[i, i]
        #x[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
    return x

def backward_sweep_GS(A, x, b,n):
    for i in range(n -1, -1, -1):
        sum = 0
        for j in range(n -1, -1, -1):
            if j != i:
                sum += A[i, j] * x[j]
        x[i] = (b[i] - sum) / A[i, i]
    return x

# Check factorization accuracy
def accuracy_decomp(A_2, M):
    sub = np.subtract(A_2, M)
    norm_num = LA.norm(sub, ord='fro')
    norm_denom = LA.norm(A_2, ord='fro')
    div = norm_num / norm_denom
    return div

# Check the solution accuracy
def accuracy_x(x, A, b):
    x_true = np.linalg.solve(A, b)
    sub = np.subtract(x_true, x)
    norm_num = LA.norm(sub, ord=2)
    norm_denom = LA.norm(x_true, ord=2)
    div = norm_num / norm_denom
    return div

# Check the accuracy via the residual
def accuracy_b(x, A_copy, b):
    b_comp = np.dot(A_copy, x)
    sub = np.subtract(b, b_comp)
    norm_num = LA.norm(sub, ord=2)
    norm_denom = LA.norm(b, ord=2)
    div = norm_num / norm_denom
    return div

# Compute the condition number
def condition_num(A_2):
    A_2_norm = np.linalg.cond(A_2)
    return A_2_norm

# Compute the growth factor
def growth_factor(A, n):
    A_abs = np.abs(A)
    A_mult = evaluate_LU(A_abs, n)
    norm_num = LA.norm(A_mult, ord='fro')
    norm_denom = LA.norm(A, ord='fro')
    div = norm_num / norm_denom
    return div
