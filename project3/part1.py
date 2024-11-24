import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1234)

"""
-------------------- Routines for Iterative Methods --------------------
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

        # Count iterations
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

        # Count Iterations
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
    err_A_norm_0 = np.sqrt(np.sum(A * (err ** 2)))
    err_arr = [err_A_norm_0]
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
        err_A_norm = np.sqrt(np.sum(A * (err ** 2)))
        err_arr.append(err_A_norm)
        err_ratio.append(err_A_norm / err_A_norm_0)

        # Keep sigma term at current step
        sigma = sigma_next

        # Count iterations
        iter += 1

    return x, iter, resid_arr, err_arr, err_ratio


"""
-------------------- Routine to Generate Test Matrices --------------------
"""
def matrix_type_diag(choice, n, lmin, lmax):
    # All eigenvalues the same
    if choice == 1:
        k = np.random.randint(lmin, lmax)
        eigenvalues = np.full(n, k)

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
-------------------- Input Collection --------------------
"""
def get_user_inputs():
    print("\nEnter range of dimensions to generate (nxn) matrix A: ")
    nmin = int(input("Minimum value: "))
    nmax = int(input("Maximum value: "))
    step = int(input("Step size: "))

    print("\nEnter range to generate random values for solution vector and initial guess vector:")
    xmin = float(input("Minimum value: "))
    xmax = float(input("Maximum value: "))

    print("\nChoose Minimum and Maximum for Eigenvalues (Must be positive): ")
    lmin = float(input("Enter lambda min: "))
    lmax = float(input("Enter lambda max: "))

    print("\nProblem types:")
    print("1. All Eigenvalues the same")
    print("2. k distinct eigenvalues with randomly chosen multiplicities")
    print("3. k distinct eigenvalues with normal distributions around each")
    print("4. Eigenvalues chosen from a Uniform Distribution, specified min lambda and max lambda")
    print("5. Eigenvalues chosen from a Normal Distribution, specified min lambda and max lambda")
    choice = int(input("Enter problem type: "))

    return nmin, nmax, step, xmin, xmax, lmin, lmax, choice


"""
-------------------- Functions for Generating Results --------------------
"""
def plot_convergence(ndim, RF_iter_avg, SD_iter_avg, CG_iter_avg, choice):
    plt.plot(ndim, RF_iter_avg, color='g', label='RF')
    plt.plot(ndim, SD_iter_avg, color='b', label='SD')
    plt.plot(ndim, CG_iter_avg, color='y', label='CG')
    plt.xlabel('Dimension n')
    plt.ylabel('Average Number of Iterations Until Convergence')
    plt.title(f'Iterations Until Convergence for Matrix Type {choice}')
    plt.legend()
    plt.savefig(f'type{choice}_iterations.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_error_ratios(ndim, error_ratios, kappa, method, choice):
    plt.figure(figsize=(8, 6))

    # Calculate the convergence bound
    if method == 'CG':
        bounds = []
        for i in range(len(error_ratios)):
            bound = (2 * (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** i
            bounds.append(bound)
        # Create the plot
        plt.plot(range(len(error_ratios)), error_ratios, label="Error Ratio", color='b')
        plt.plot(range(len(error_ratios)), bounds, linestyle='--', color='r')

    else:
        bound = (kappa - 1) / (kappa + 1)
        if choice == 1:
            bound = 1
        # Create the plot
        plt.plot(range(len(error_ratios)), error_ratios, label="Error Ratio", color='b')
        plt.axhline(y=bound, linestyle='--', color='r')

    # Add labels, title, and legend
    plt.xlabel("Iterations")
    plt.ylabel("Error Ratio")
    plt.title(f"Error Ratio for {method} (n = {ndim})")
    plt.legend()
    plt.grid(True)

    # Save and show the plot
    plt.savefig(f'error_ratios_{method}_{choice}_{ndim}.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_errs_and_resids(ndim, err_array, resid_array, method, choice):
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(err_array)), err_array, label='Error Terms', color='y')
    plt.plot(range(len(resid_array)), resid_array, label='Residual Terms', color='g')

    # Add labels, title, and legend
    plt.xlabel("Iterations")
    plt.ylabel("Errors and Residuals")
    plt.title(f"Convergence of Error and Residual Terms for {method} (n = {ndim})")
    plt.legend()
    plt.grid(True)

    # Save and show the plot
    plt.savefig(f'errrors_and_residuals_{method}_{choice}_{ndim}.png', dpi=300, bbox_inches='tight')
    plt.show()


"""
-------------------- Main Routine --------------------
"""
# nmin, nmax, step, xmin, xmax, lmin, lmax, choice = get_user_inputs()
ndim = [10, 50, 100]
xmin = -100
xmax = 100
lmin = 1
lmax = 100
RF_iter_avg = []
SD_iter_avg = []
CG_iter_avg = []

for choice in range(1,6):
    for n in ndim:
        A = matrix_type_diag(choice, n, lmin, lmax)
        x_tilde = np.random.uniform(xmin, xmax, n)
        x0 = np.random.uniform(xmin, xmax, n)
        b = A * x_tilde
        kappa = np.max(A) / np.min(A)


        x1, iter1, resid_arr1, err_arr1, err_ratio1 = richardsons_stationary(A, x_tilde, x0, b)
        plot_error_ratios(n, err_ratio1, kappa, 'RF', choice)
        plot_errs_and_resids(n, err_arr1, resid_arr1, 'RF', choice)
        x2, iter2, resid_arr2, err_arr2, err_ratio2 = steepest_descent(A, x_tilde, x0, b)
        plot_error_ratios(n, err_ratio2, kappa, 'SD', choice)
        plot_errs_and_resids(n, err_arr2, resid_arr2, 'SD', choice)
        x3, iter3, resid_arr3, err_arr3, err_ratio3 = conjugate_gradient(A, x_tilde, x0, b)
        plot_error_ratios(n, err_ratio3, kappa, 'CG', choice)
        plot_errs_and_resids(n, err_arr3, resid_arr3, 'CG', choice)


"""for n in range(nmin, nmax + 1, step):
    RF_iter = []
    SD_iter = []
    CG_iter = []

    ndim.append(n)
    A = matrix_type_diag(choice, n, lmin, lmax)

    for i in range(5):
        x_tilde = np.random.uniform(xmin, xmax, n)
        x0 = np.random.uniform(xmin, xmax, n)
        b = A * x_tilde
        x1, iter1, resid_arr1, err_arr1, err_ratio1 = richardsons_stationary(A, x_tilde, x0, b)
        x2, iter2, resid_arr2, err_arr2, err_ratio2 = steepest_descent(A, x_tilde, x0, b)
        x3, iter3, resid_arr3, err_arr3, err_ratio3 = conjugate_gradient(A, x_tilde, x0, b)
        RF_iter.append(iter1)
        SD_iter.append(iter2)
        CG_iter.append(iter3)

    RF_iter_avg.append(np.average(RF_iter))
    SD_iter_avg.append(np.average(SD_iter))
    CG_iter_avg.append(np.average(CG_iter))

plot_convergence(ndim, RF_iter_avg, SD_iter_avg, CG_iter_avg, choice)"""









