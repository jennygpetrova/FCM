import numpy as np

"""
-------------------- Functions for Iterative Methods --------------------
"""

# Richardson's First Order Stationary Method
def richardsons_stationary(A, x_tilde, x0, b):
    # Optimal alpha for diagonal matrix
    alpha = 2 / (np.max(A) + np.min(A))

    # Initial residual
    r = b - A * x0
    r0_norm = np.linalg.norm(r)
    resid_arr = [r0_norm]
    r_norm = r0_norm

    # Initial error
    err = x0 - x_tilde
    err_arr = [np.linalg.norm(err)]
    err_ratio = [1.0]

    # Initial guess
    x = x0.copy()

    # Termination criterion
    max_iter = 1000
    tol = 1e-6
    iter = 0

    while iter < max_iter and r_norm / r0_norm > tol:
        # Iteration updates
        x += alpha * r
        r -= alpha * (A * r)

        # Store residual norms
        r_norm = np.linalg.norm(r)
        resid_arr.append(r_norm)

        # Store error
        err_next = x - x_tilde
        err_arr.append(np.linalg.norm(err_next))
        err_ratio.append(np.linalg.norm(err_next) / np.linalg.norm(err))

        # Keep error term at current step after calculating error ratio
        err = err_next

        # Increment iteration counter
        iter += 1

    return x, iter, resid_arr, err_arr, err_ratio


# Steepest Descent Method (SD)
def steepest_descent(A, x_tilde, x0, b):
    # Initial residual
    r = b - A * x0
    r0_norm = np.linalg.norm(r)
    resid_arr = [r0_norm]
    r_norm = r0_norm

    # Initial error
    err = x0 - x_tilde
    err_A_norm = np.sqrt(np.sum(A * (err ** 2)))
    err_arr = [err_A_norm]
    err_ratio = [1.0]

    # Initial guess
    x = x0.copy()

    # Termination criterion
    max_iter = 1000
    tol = 1e-6
    iter = 0

    while iter < max_iter and r_norm / r0_norm > tol:
        # Iteration updates
        v = A * r
        alpha = np.dot(r, r) / np.dot(r, v)
        x += alpha * r
        r -= alpha * v

        # Store residual norms
        r_norm = np.linalg.norm(r)
        resid_arr.append(r_norm)

        # Store error
        err = x - x_tilde
        err_A_norm_next = np.sqrt(np.sum(A * (err ** 2)))
        err_arr.append(err_A_norm_next)
        err_ratio.append(err_A_norm_next / err_A_norm)

        # Keep error term at current step after calculating error ratio
        err_A_norm = err_A_norm_next

        # Increment iteration counter
        iter += 1

    return x, iter, resid_arr, err_arr, err_ratio


# Conjugate Gradient Method (CG)
def conjugate_gradient(A, x_tilde, x0, b):
    # Initial residual
    r = b - A * x0
    r0_norm = np.linalg.norm(r)
    resid_arr = [r0_norm]
    r_norm = r0_norm

    # Initial error
    err = x0 - x_tilde
    err_A_norm = np.sqrt(np.sum(A * (err ** 2)))
    err_arr = [err_A_norm]
    err_ratio = [1.0]

    # Initial variables
    x = x0.copy()
    d = r.copy()
    sigma = np.dot(r, r)

    # Termination criterion
    max_iter = 1000
    tol = 1e-6
    iter = 0

    while iter < max_iter and r_norm / r0_norm > tol:
        # Iteration updates
        v = A * d
        mu = np.dot(d, v)
        alpha = sigma / mu
        x += alpha * d
        r -= alpha * v
        sigma_next = np.dot(r, r)
        beta = sigma_next / sigma
        d = r + beta * d

        # Store residual norms
        r_norm = np.linalg.norm(r)
        resid_arr.append(r_norm)

        # Store error
        err = x - x_tilde
        err_A_norm_next = np.sqrt(np.sum(A * (err ** 2)))
        err_arr.append(err_A_norm_next)
        err_ratio.append(err_A_norm_next / err_A_norm)

        # Keep error term and sigma term at current step
        err_A_norm = err_A_norm_next
        sigma = sigma_next

        iter += 1

    return x, iter, resid_arr, err_arr, err_ratio

"""
-------------------- Routine to Generate Test Matrices --------------------
"""
def part_1_driver(choice, n, lmin, lmax):
    # All eigenvalues the same
    if choice == 1:
        eigenvalues = np.full(n, 10)

    # k distinct eigenvalues with randomly chosen multiplicities
    elif choice == 2:
        k = np.random.randint(1, n + 1)
        lambdas = np.random.uniform(lmin, lmax, k)
        multiplicities = np.random.multinomial(n, [1 / k] * k)
        eigenvalues = np.repeat(lambdas, multiplicities)

    # k random eigenvalues from a cloud of normally distributed eigenvalues
    elif choice == 3:
        k = np.random.randint(1, n + 1)
        lambdas = np.random.uniform(lmin, lmax, k)
        multiplicities = np.random.multinomial(n, [1 / k] * k)
        eigenvalues = []
        for l, m in zip(lambdas, multiplicities):
            cloud = np.random.normal(l, 1, m)
            eigenvalues.extend(cloud)
        eigenvalues = np.array(eigenvalues)

    # n uniformly distributed eigenvalues
    elif choice == 4:
        eigenvalues = np.random.uniform(lmin, lmax, n)

    # n normally distributed eigenvalues
    else:
        mean = (lmin + lmax) / 2
        std_dev = (lmax - lmin) / 6
        eigenvalues = np.random.normal(mean, std_dev, n)

    return np.array(eigenvalues)

"""
-------------------- Driver --------------------
"""
def get_user_inputs():
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
RF_ratio = []

SD_x = []
SD_iter = []
SD_res = []
SD_err = []
SD_ratio = []

CG_x = []
CG_iter = []
CG_res = []
CG_err = []
CG_ratio = []

for n in range(nmin, nmax + 1):
    ndim.append(n)
    x_tilde = np.random.uniform(dmin, dmax, n)
    x0 = np.random.uniform(dmin, dmax, n)

    print("\nSolution Vector x_tilde = ", x_tilde)
    print("\nInitial Guess Vector x0 = ", x0)

    # Generate diagonal matrix stored as a vector of eigenvalues
    A = part_1_driver(choice, n, lmin, lmax)
    print("\nDiagonal Matrix A (as vector): ", A)

    # Compute b_tilde
    b_tilde = A * x_tilde
    print("\nb_tilde = A * x_tilde: ", b_tilde)

    # Richardson's Stationary Method
    x_RF, iter_RF, resid_arr_RF, err_arr_RF, err_ratio_RF = richardsons_stationary(A, x_tilde, x0, b_tilde)
    print("\nRF Solution Vector: ", x_RF)
    print("Number of Iterations (RF): ", iter_RF)
    RF_iter.append(iter_RF)

    # Steepest Descent Method
    x_SD, iter_SD, resid_arr_SD, err_arr_SD, err_ratio_SD = steepest_descent(A, x_tilde, x0, b_tilde)
    print("\nSD Solution Vector: ", x_SD)
    print("Number of Iterations (SD): ", iter_SD)
    SD_iter.append(iter_SD)

    # Conjugate Gradient Method
    x_CG, iter_CG, resid_arr_CG, err_arr_CG, err_ratio_CG = conjugate_gradient(A, x_tilde, x0, b_tilde)
    print("\nCG Solution Vector: ", x_CG)
    print("Number of Iterations (CG): ", iter_CG)
    CG_iter.append(iter_CG)

# Compute condition number and convergence bounds
kappa = np.max(A) / np.min(A)
bound_RF_SD = (kappa - 1) / (kappa + 1)
bound_CG = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)
print(f"\nCondition Number (kappa): {kappa:.2f}")
print(f"Bound for RF/SD Convergence Rate: {bound_RF_SD:.4f}")
print(f"Bound for CG Convergence Rate: {bound_CG:.4f}")



"""
-------------------- Plots and Graphs --------------------
"""
