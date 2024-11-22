import numpy as np
import matplotlib as plt

"""
-------------------- Routine for Generating a Sparse Matrix and Storing in CSR Format --------------------
"""
# Sparse symmetric (diagonally dominant) matrix
def sparse_matrix(n):
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i > j:
                x = np.random.choice([0,1])
                if x == 1:
                    # Scale each row to ensure diagonal dominance
                    A[i][j] = np.round(np.random.randint(1,10) / i, 3)
    A = A + A.T
    np.fill_diagonal(A, 10)
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

# Matrix-vector multiplication when A is stored as CSR
def csr_multiply(AA, JA, IA, x):
    b = np.zeros(len(IA) - 1)  # Result vector
    for i in range(len(b)):
        for k in range(IA[i], IA[i + 1]):
            b[i] += AA[k] * x[JA[k]]
    return b

A = sparse_matrix(5)
print(A)
x = np.random.randint(1,10,5)
mult = np.dot(A,x)
print(mult)
AA, JA, IA = compressed_row(A)
print(AA)
print(JA)
print(IA)
b = csr_multiply(AA, JA, IA, x)
print(b)
