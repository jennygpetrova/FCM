import warnings
import numpy as np
from numpy import linalg as LA
import scipy
from scipy import linalg
np.random.seed(42)

#Function for factorization routine
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
def x_solve(A, b, n):
    # Forward substitution to solve Ly = b for y (where L is stored in A)
    y = np.zeros(n)
    y[0] = b[0]
    for i in range(1, n):
        y[i] = b[i] - np.dot(A[i, :i], y[:i])

    # Backward substitution to solve Ux = y for x (where U is stored in A)
    x = np.zeros(n)
    x[n - 1] = y[n - 1] / A[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (y[i] - np.dot(A[i, i + 1:n], x[i + 1:n])) / A[i, i]

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

if __name__ == '__main__':
    n = int(input("Enter Dimension for Matrix: "))
    A = np.zeros((n, n))

    # Randomly generated conditioned Matrix A with specified L and U
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j > i:
                U[i, j] = np.random.uniform(-10, 10)
            if j == i:
                U[i, j] = np.random.uniform(-10, 10)
                L[i, j] = 1
            if j < i:
                L[i,j] = np.random.uniform(-1, 1)
    print("L \n:", L)
    print("U \n:", U)
    A = np.dot(L, U)
    print("A \n:", A)
    #A = np.random.uniform(1, 10, (n, n)).astype(float) # Generate a random nxn 2D-array (matrix) A
    #print("{}x{} Matrix A: \n".format(n, n), A)

    # Matrices for Empirical Tasks
        # Comment out the randomly generated matrix A above
        # Uncomment the matrices below for the specific matrix A you want to test
    # (1)
    # for i in range(n):
    #    A[i, i] = i+1
    # (2)
    # for i in range(n):
    #    A[i, n-i-1] = i+1
    # (3)
    # for i in range(n):
    #    A[i, i] = i+1
    #    A[i, n-i-1] += i+1
    # (4)
    # for i in range(n):
    #    for j in range(n):
    #        if i > j:
    #            A[i][j] = np.random.uniform(-1, 1)
    #        if i == j:
    #            A[j][i] = 1
    #        if i < j:
    #            A[j][i] = 0
    # (5)
    # a = np.random.randint(2, 10)
    # for i in range(n):
    #    for j in range(n):
    #        if i > j:
    #            A[i][j] = np.random.randint(2, 10)
    #        if i == j:
    #            A[j][i] = a
    #        if i < j:
    #            A[j][i] = 0
    # (6)
    # subdiag = np.random.uniform(1, 5, size=n-1)
    # superdiag = np.random.uniform(1,5, size=n-1)
    # diag = np.random.uniform(10,20, size=n)
    # for i in range(n):
    #    A[i,i] = diag[i]
    #    if i < n-1:
    #        A[i+1][i] = subdiag[i]
    #        A[i][i+1] = superdiag[i]
    # (7)
    # for i in range(n):
    #    for j in range(n):
    #        if i == j:
    #            A[i, j] = 1
    #        elif i > j:
    #            A[i, j] = -1
    #        elif j == n-1:
    #            A[i, j] = 1
    # (8)
    # L_tilde = np.zeros((n, n))
    # for i in range(n):
    #    for j in range(n):
    #        if i >= j:
    #            L_tilde = np.tril(np.random.uniform(-1, 1, (n, n)))
    # np.fill_diagonal(L_tilde, np.random.uniform(1, 2, n))
    # A = np.dot(L_tilde, L_tilde.T)
    #print("L̃ (Lower triangular matrix):")
    #print(L_tilde)
    #print("L̃ Transpose:")
    #print(np.transpose(L_tilde))
    #print("\nA (Symmetric Positive Definite matrix):")
    #print(A)

    print("Form of Factorization: \n (1) No Pivoting \n (2) Partial Pivoting \n (3) Complete Pivoting")
    flag = int(input("Enter Factorization Type: "))

    A_copy = np.copy(A) #Create a copy of A for correctness checking

    Pr, Pc, A = LU_decomp(A, n, flag)
    print("Row Permutation Array P: \n", Pr)
    print("Column Permutation Array P: \n", Pc)
    print("A Factorized into LU: \n", A)

    M = evaluate_LU(A, n)
    print("Result of L*U: \n", M)

    A_2 = compute_PrAPc(A_copy, Pr, Pc)
    print("Result of Pr*A*Pc:\n", A_2)

    b = np.random.randint(1, 10, (n)).astype(float) # Generate a random nx1 1D-array b
    print("Randomly Generated vector b: \n", b)

    # SciPy LU decomposition for true solution when partial pivoting
    # Used to check accuracy while building code, has no further applications in the routine
    if flag == 2:
        P_scipy, L_scipy, U_scipy = scipy.linalg.lu(A_copy)
        print("SciPy LU Decomposition (with Partial Pivoting)")
        print("Permutation Matrix P:\n", P_scipy)
        print("Lower Triangular Matrix L:\n", L_scipy)
        print("Upper Triangular Matrix U:\n", U_scipy)

    b = b[Pr]  # Apply Pr to b before checking accuracies (true solution for b will be permuted)
    x = x_solve(A, b, n)

    print("Forward and Backward solve Ax=b for x: \n", x)

    print("Factorization accuracy:\n", accuracy_decomp(A_2, M))
    print("Accuracy via x:\n", accuracy_x(x, A_2, b))
    print("Accuracy via b:\n", accuracy_b(x, A_2, b))
    print("Condition Number:\n", condition_num(A_copy))
    print("Growth Factor:\n", growth_factor(A, n))


