import numpy as np
from project2 import LU_factorization as lu
import matplotlib.pyplot as plt
import random
np.random.seed(42)

nmin = int(input("Min Dimensions to Test: "))
nmax = int(input("Max Dimensions to Test: "))
ndelta = int(input("Steps in Dimension Range: "))
ntests = int(input("Number of Tests per Dimension: "))
print("Form of Factorization: \n (1) No Pivoting \n (2) Partial Pivoting \n (3) Complete Pivoting")
flag = int(input("Enter Factorization Type: "))

dmin = -10
dmax = 10
mean = 0
std = 500
AError2norm = [0] * ntests
xtrue2Norm = [0] * ntests
btrue2Norm = [0] * ntests
cond_num = [0] * ntests
growth_factor =  [0] * ntests
nexperiments = len(range(nmin, nmax + 1, ndelta))
Ameanforthisn = [0] * nexperiments
Amaxforthisn =  [0] * nexperiments
xmeanforthisn = [0] * nexperiments
xmaxforthisn =  [0] * nexperiments
bmeanforthisn = [0] * nexperiments
bmaxforthisn =  [0] * nexperiments
cmeanforthisn = [0] * nexperiments
cmaxforthisn = [0] * nexperiments
gmeanforthisn = [0] * nexperiments
gmaxforthisn = [0] * nexperiments
value_of_n = [0] * nexperiments

k = 0

for i in range(nmin, nmax+1, ndelta):
    print("{}x{} Matrix A:".format(i, i))
    value_of_n[k] = i
    for j in range(ntests):
        L = np.zeros((i, i))
        U = np.zeros((i, i))
        scale_factor = 1/i
        for a in range(i):
            for b in range(i):
                if b > a:
                    U[a, b] = np.random.uniform(-10, 10) * scale_factor
                if b == a:
                    if random.choice([True, False]):
                        U[a, b] = random.uniform(5, 10)
                    else:
                        U[a, b] = random.uniform(-10, -5)
                    L[a, b] = 1
                if b < a:
                    L[a, b] = np.random.uniform(-1, 1) * scale_factor
        A = np.dot(L, U)
        P_rand = np.arange(i)
        random.shuffle(P_rand)
        P_rand2 = np.arange(i)
        random.shuffle(P_rand2)
        A_perm = A[P_rand, :]
        A_perm2 = A_perm[:, P_rand2]
        A = np.copy(A_perm2)
        #A = np.random.normal(mean, std, (i, i))
        A_copy = np.copy(A)
        Pr, Pc, A = lu.LU_decomp(A, i, flag)
        M = lu.evaluate_LU(A, i)
        A_2 = lu.compute_PrAPc(A_copy, Pr, Pc)
        b = np.random.uniform(dmin, dmax, (i))
        #b = np.random.normal(mean, std, (i))
        b = b[Pr]
        x = lu. x_solve(A, b, i)
        AError2norm[j] = lu.accuracy_decomp(A_2, M)
        xtrue2Norm[j] = lu.accuracy_x(x, A_2, b)
        btrue2Norm[j] = lu.accuracy_b(x, A_2, b)
        cond_num[j] = lu.condition_num(A_2)
        growth_factor[j] = lu.growth_factor(A, i)
    Ameanforthisn[k] = np.mean(AError2norm)
    print("Avg Factorization Error: ", np.mean(AError2norm))
    Amaxforthisn[k] = np.max(AError2norm)
    bmeanforthisn[k] = np.mean(btrue2Norm)
    print("Avg b Error: ", np.mean(btrue2Norm))
    bmaxforthisn[k] = np.max(btrue2Norm)
    xmeanforthisn[k] = np.mean(xtrue2Norm)
    print("Avg x Error: ", np.mean(xtrue2Norm))
    xmaxforthisn[k] = np.max(xtrue2Norm)
    cmeanforthisn[k] = np.mean(cond_num)
    print("Avg Condition Number: ", np.mean(cond_num))
    cmaxforthisn[k] = np.max(cond_num)
    print("Max Condition Number: ", np.max(cond_num))
    gmeanforthisn[k] = np.mean(growth_factor)
    print("Avg Growth Factor: ", np.mean(growth_factor))
    gmaxforthisn[k] = np.max(growth_factor)
    print("Max Growth Factor: ", np.max(growth_factor))
    k += 1

print("Average b Error Across All Tests: ", np.mean(bmeanforthisn))
print("Average x Error Across All Tests: ", np.mean(bmeanforthisn))

plt.scatter(value_of_n, Ameanforthisn, marker='o', color='g', label='A Mean Error')
plt.scatter(value_of_n, Amaxforthisn, marker='o', color='y', label='A Max Error')
plt.xlabel('Dimension n')
plt.ylabel('Relative Error of Factorization of A')
plt.title('Factorization Accuracy per Dimension Tested (Complete Pivoting)')
plt.legend()
plt.show()
plt.savefig('A_cc.png')

plt.scatter(value_of_n, xmeanforthisn, marker='o', color='b', label='x Mean Error')
plt.scatter(value_of_n, xmaxforthisn, marker='o', color='r', label='x Max Error')
plt.xlabel('Dimension n')
plt.ylabel('Relative Error of x')
plt.title('Accuracy of Solution per Dimension Tested (Complete Pivoting)')
plt.legend()
plt.show()
plt.savefig('x_cc.png')

plt.scatter(value_of_n, bmeanforthisn, marker='o', color='b', label='b Mean Error')
plt.scatter(value_of_n, bmaxforthisn, marker='o', color='r', label='b Max Error')
plt.xlabel('Dimension n')
plt.ylabel('Relative Error of b')
plt.title('Accuracy via Residuals per Dimension Tested (Complete Pivoting)')
plt.legend()
plt.show()
plt.savefig('b_cc.png')

plt.scatter(value_of_n, cmeanforthisn, marker='o', color='c', label='Mean Cond Num')
plt.scatter(value_of_n, cmaxforthisn, marker='o', color='m', label='Max Cond Num')
plt.xlabel('Dimension n')
plt.ylabel('Condition Number')
plt.title('Numerical Stability of Matrices per Dimension Tested (Complete Pivoting)')
plt.legend()
plt.show()
plt.savefig('c_cc.png')

plt.scatter(value_of_n, gmeanforthisn, marker='o', color='c', label='Mean Growth Factor')
plt.scatter(value_of_n, gmaxforthisn, marker='o', color='m', label='Max Growth Factor')
plt.xlabel('Dimension n')
plt.ylabel('Growth Factor')
plt.title('Numerical Stability of Factorization per Dimension Tested (Complete Pivoting)')
plt.legend()
plt.show()
plt.savefig('g_cc.png')