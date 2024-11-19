import warnings
import numpy as np
import random
random.seed(123)

n = int(input("Enter Dimension for Matrix: ")) #Input the dimension n

#Matrices for Empirical Tasks
A = np.zeros((n, n)) #Remove the hashtags below for the matrix you want to test
#(1)
#for i in range(n):
#    A[i, i] = i+1
#(2)
#for i in range(n):
#    A[i, n-i-1] = i+1
#(3)
#for i in range(n):
#    A[i, i] = i+1
#    A[i, n-i-1] += i+1
#(4)
#for i in range(n):
#    for j in range(n):
#        if i > j:
#            A[i][j] = np.random.uniform(-1, 1)
#        if i == j:
#            A[j][i] = 1
#        if i < j:
#            A[j][i] = 0
#(5)
#a = np.random.randint(2, 10)
#for i in range(n):
#    for j in range(n):
#        if i > j:
#            A[i][j] = np.random.randint(2, 10)
#        if i == j:
#            A[j][i] = a
#        if i < j:
#            A[j][i] = 0
#(6)
#subdiag = np.random.uniform(1, 5, size=n-1)
#superdiag = np.random.uniform(1,5, size=n-1)
#diag = np.random.uniform(10,20, size=n)
#for i in range(n):
#    A[i,i] = diag[i]
#    if i < n-1:
#        A[i+1][i] = subdiag[i]
#        A[i][i+1] = superdiag[i]
#(7)
#for i in range(n):
#    for j in range(n):
#        if i == j:
#            A[i, j] = 1
#        elif i > j:
#            A[i, j] = -1
#        elif j == n-1:
#            A[i, j] = 1
#(8)
#L_tilde = np.zeros((n, n))
#for i in range(n):
#    for j in range(n):
#        if i >= j:
#            L_tilde[i, j] = np.random.uniform(1, 5)
#A = np.dot(L_tilde, L_tilde.T)

print("{}x{} Matrix A: \n".format(n, n), A)

print("Form of Factorization: \n (1) No Pivoting \n (2) Partial Pivoting \n (3) Complete Pivoting") #Pick a flag for no pivoting or partial pivoting
flag = int(input("Enter Factorization Type: "))

A_copy = np.copy(A) #Create a copy of A for correctness checking

#Function for factorization routine
def LU_decomp(A, n, flag):
    Pr = np.arange(n)  # 1D-array to store row permutations
    Pc = np.arange(n)  # 1D-array to store column permutations (used in complete pivoting)

    for i in range(n):
        if flag == 1:  # No pivoting
            if np.abs(A[i][i]) < 1e-10:  # Detect where factorization may not proceed
                warnings.warn("({},{}) element too small".format(i, i), UserWarning)
                return None, None  # Factorization should not proceed

        elif flag == 2:  # Partial pivoting (row pivoting)
            max_row = np.argmax(np.abs(A[i:, i])) + i  # Find row with max element in column i
            if i != max_row:
                A[[i, max_row], :] = A[[max_row, i], :]  # Swap rows in A
                Pr[[i, max_row]] = Pr[[max_row, i]]  # Track row swaps in permutation array

            if np.all(np.abs(A[i:, i]) < 1e-10):  # Detect where factorization may not proceed
                warnings.warn("({},{}) column has all elements too small.".format(i, i), UserWarning)
                return None, None  # Factorization should not proceed

        elif flag == 3:  # Complete pivoting (row and column pivoting)
            # Find the location of the maximum element in submatrix A[i:, i:]
            max_row, max_col = divmod(np.argmax(np.abs(A[i:, i:])), n - i)
            max_row += i
            max_col += i

            if np.abs(A[max_row, max_col]) < 1e-10:  # If the max element is too small
                warnings.warn(f"({max_row},{max_col}) element too small", UserWarning)
                return None, None, None  # Factorization should not proceed

            # Swap rows
            if max_row != i:
                A[[i, max_row], :] = A[[max_row, i], :]
                Pr[[i, max_row]] = Pr[[max_row, i]]  # Track row swaps

            # Swap columns
            if max_col != i:
                A[:, [i, max_col]] = A[:, [max_col, i]]
                Pc[[i, max_col]] = Pc[[max_col, i]]  # Track column swaps

        # Perform LU factorization for the current column
        for j in range(i+1, n):
            lam = A[j][i] / A[i][i]  # Compute lambda
            A[j][i] = lam  # Store lambda below the diagonal for L
            for k in range(i+1, n):
                A[j][k] -= lam * A[i][k]  # Update U entries

    # Return the permutation arrays and the modified matrix
    if flag == 2:
        return Pr, A  # Return row permutations and factorized matrix
    elif flag == 3:
        return Pr, Pc, A  # Return row and column permutations for complete pivoting
    else:
        return A

if flag == 3:
    Pr, Pc, A = LU_decomp(A, n, flag)
    print("Row Permutation Array P: \n", Pr)
    print("Column Permutation Array P: \n", Pc)
elif flag == 2:
    Pr, A = LU_decomp(A, n, flag)
    print("Row Permutation Array P: \n", Pr)
else:
    A = LU_decomp(A, n, flag)

print("L and U Computed In-Place of Matrix A: \n", A)


