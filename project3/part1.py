from mypackage import myfunctions
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1234)

"""
-------------------- Functions for Iterative Methods --------------------
"""

# Richardson's First Order Stationary Method
def richardsons_stationary(A, x_tilde, x0):
    alpha = 2 / (np.max(A) + np.min(A))  # Optimal alpha for diagonal matrix
    b_tilde = A * x_tilde  # A is a vector, so use element-wise multiplication

    r = b_tilde - A * x0  # Initial residual
    r0_norm = np.linalg.norm(r)
    resid_arr = [r0_norm]

    err = np.subtract(x_tilde, x0)  # Initial error
    err_arr = [np.linalg.norm(err)]

    x = x0

    # Termination criterion
    max_iter = 1000
    tol = 1e-6
    iter = 0

    for _ in range(max_iter):
        x += alpha * r  # Update x
        r = b_tilde - A * x  # Update residual

        r_norm = np.linalg.norm(r)
        resid_arr.append(r_norm)

        err = np.subtract(x_tilde, x)
        err_arr.append(np.linalg.norm(err))

        if r_norm / r0_norm < tol:
            break

        iter += 1

    return x, iter, resid_arr, err_arr


# Steepest Descent Method (SD)
def steepest_descent(A, x_tilde, x0):
    b_tilde = A * x_tilde  # A is a vector, so use element-wise multiplication

    r = b_tilde - A * x0  # Initial residual
    r0_norm = np.linalg.norm(r)
    resid_arr = [r0_norm]

    err = np.subtract(x_tilde, x0)  # Initial error
    err_arr = [np.linalg.norm(err)]

    x = x0

    max_iter = 1000
    tol = 1e-6
    iter = 0

    for _ in range(max_iter):
        v = A * r  # Element-wise multiplication
        alpha = np.dot(r, r) / np.dot(r, v)  # Calculate step size alpha

        x += alpha * r  # Update x
        r -= alpha * v  # Update residual

        r_norm = np.linalg.norm(r)
        resid_arr.append(r_norm)

        err = np.subtract(x_tilde, x)  # Update error
        err_arr.append(np.linalg.norm(err))

        if r_norm / r0_norm < tol:  # Check termination criterion
            break

        iter += 1

    return x, iter, resid_arr, err_arr


# Conjugate Gradient Method (CG)
def conjugate_gradient(A, x_tilde, x0):
    b_tilde = A * x_tilde  # A is a vector, so use element-wise multiplication

    r = b_tilde - A * x0
    r0_norm = np.linalg.norm(r)
    resid_arr = [r0_norm]

    err = np.subtract(x_tilde, x0)
    err_arr = [np.linalg.norm(err)]

    x = x0
    d = r
    sigma = np.dot(r, r)

    max_iter = 1000
    tol = 1e-6
    iter = 0
    for _ in range(max_iter):
        v = A * d  # Element-wise multiplication
        mu = np.dot(d, v)
        alpha = sigma / mu

        x += alpha * d
        r -= alpha * v
        sigma_next = np.dot(r, r)

        beta = sigma_next / sigma
        d = r + beta * d

        r_norm = np.linalg.norm(r)
        resid_arr.append(r_norm)

        err = np.subtract(x_tilde, x)
        err_arr.append(np.linalg.norm(err))

        if r_norm / r0_norm < tol:
            break

        iter += 1
        sigma = sigma_next

    return x, iter, resid_arr, err_arr


"""
-------------------- Driver to Test Matrices --------------------
"""
def part_1_driver(choice, n, lmin, lmax):
    # All eigenvalues the same
    if choice == 1:
        eigenvalues = np.full(n, 10)

    # k distinct eigenvalues with randomly chosen multiplicities
    elif choice == 2:
        k = np.random.randint(1, n + 1)
        lambdas = np.random.choice(np.arange(1, n + 1), size=k, replace=False)
        multiplicities = np.random.multinomial(n, [1 / k] * k)
        eigenvalues = []
        for l, m in zip(lambdas, multiplicities):
            eigenvalues.extend([l] * m)

    # k random eigenvalues
    elif choice == 3:
        k = np.random.randint(1, n + 1)
        lambdas = np.random.choice(np.arange(1, n + 1), size=k, replace=False)
        multiplicities = np.random.multinomial(n, [1 / k] * k)
        eigenvalues = []
        for l, m in zip(lambdas, multiplicities):
            cloud = np.random.normal(l, 1, m)
            eigenvalues.extend(cloud)

    # n uniformly distributed eigenvalues
    elif choice == 4:
        eigenvalues = np.random.uniform(lmin, lmax, n)

    # n normally distributed eigenvalues
    else:
        mean = (lmin - lmax) / 2
        eigenvalues = np.random.normal(mean, 1, n)

    return eigenvalues


"""
Run the 3 methods for each choice of A, for several values of A for several sizes n.
"""
def get_user_inputs():
    """
    Prompts user for:
        - Size of the (nxn) matrix A
        - Range of random values (xmin, xmax) to generate solution vector x_tilde
        - Minimum and maximum eigenvalues (lmin, lmax)
    """

    print("\nEnter range of dimensions to generate (nxn) matrix A: ")
    nmin = int(input("Minimum value: "))
    nmax = int(input("Maximum value: "))
    print("\nEnter range to generate random values for solution vector and initial guess vector:")
    dmin = float(input("Minimum value: "))
    dmax = float(input("Maximum value: "))
    print("\nChoose Minimum and Maximum for Eigenvalues (Must be positive): ")
    lmin = float(input("Enter lambda min: "))
    lmax = float(input("Enter lambda max: "))

    print("\nProblem types:")
    print("1. All Eigenvalues the same")
    print("2. k distinct eigenvalues with multiplicities")
    print("3. k distinct eigenvalues with random distributions around each")
    print("4. Eigenvalues chosen from a Uniform Distribution, specified min lambda and max lambda")
    print("5. Eigenvalues chosen from a Normal Distribution, specified min lambda and max lambda")
    choice = int(input("Enter problem type: "))

    return nmin, nmax, dmin, dmax, lmin, lmax, choice

nmin, nmax, dmin, dmax, lmin, lmax, choice = get_user_inputs()

ndim = []

RF_x = []
RF_iter = []
RF_res = []
RF_err = []

SD_x = []
SD_iter = []
SD_res = []
SD_err = []

CG_x = []
CG_iter = []
CG_res = []
CG_err = []

for n in range(nmin, nmax+1, 2):
    ndim.append(n)
    x_tilde = np.random.uniform(dmin, dmax, n)
    x0 = np.random.uniform(dmin, dmax, n)

    print("\nSolution Vector x_tilde = ", x_tilde)
    print("\nInitial Guess Vector x_0 = ", x0)
    A = part_1_driver(choice, n, lmin, lmax)
    print("\nDiagonal Matrix A: ", A)
    b_tilde = np.dot(A, x_tilde)

    x_RF, iter_RF, resid_arr_RF, err_arr_RF = richardsons_stationary(A, x_tilde, x0)
    print("\nRF solution vector: ", x_RF)
    print("\nNumber of iterations: ", iter_RF)
    RF_err.append(err_arr_RF[-1])
    RF_iter.append(iter_RF)

    x_SD, iter_SD, resid_arr_SD, err_arr_SD = steepest_descent(A, x_tilde, x0)
    print("\nSD stationary solution vector: ", x_SD)
    print("\nNumber of iterations: ", iter_SD)
    SD_err.append(err_arr_SD[-1])
    SD_iter.append(iter_RF)

    x_CG, iter_CG, resid_arr_CG, err_arr_CG = conjugate_gradient(A, x_tilde, x0)
    print("\nCG solution vector: ", x_CG)
    print("\nNumber of iterations: ", iter_CG)
    CG_err.append(err_arr_CG[-1])
    CG_iter.append(iter_RF)

    kappa = np.max(A)/np.min(A)
    bound_RF_SD = (kappa - 1)/(kappa + 1)
    bound_CG = (np.sqrt(kappa) - 1)/(np.sqrt(kappa) + 1)


""" 
-------------------- Plot Results for Each Method --------------------
"""
