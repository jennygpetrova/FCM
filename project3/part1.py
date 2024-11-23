import numpy as np
import matplotlib.pyplot as plt
import time
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
-------------------- Input Collection --------------------
"""
def get_user_inputs():
    print("\nEnter range of dimensions to generate (nxn) matrix A: ")
    nmin = int(input("Minimum value: "))
    nmax = int(input("Maximum value: "))

    print("\nEnter range to generate random values for solution vector and initial guess vector:")
    xmin = float(input("Minimum value: "))
    xmax = float(input("Maximum value: "))

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

    return nmin, nmax, xmin, xmax, lmin, lmax, choice

"""
-------------------- Helper Functions --------------------
"""
def run_methods(A, x_tilde, x0, b_tilde):
    # Run all methods and collect results
    results = {}
    results['RF'] = richardsons_stationary(A, x_tilde, x0, b_tilde)
    results['SD'] = steepest_descent(A, x_tilde, x0, b_tilde)
    results['CG'] = conjugate_gradient(A, x_tilde, x0, b_tilde)
    return results

def plot_convergence(ndim, RF_iter_avg, SD_iter_avg, CG_iter_avg, choice):
    # Compare iterations until convergence
    plt.plot(ndim, RF_iter_avg, color='g', label='RF')
    plt.plot(ndim, SD_iter_avg, color='b', label='SD')
    plt.plot(ndim, CG_iter_avg, color='y', label='CG')
    plt.xlabel('Dimension n')
    plt.ylabel('Number of Iterations Until Convergence')
    plt.title(f'Iterations Until Convergence for Matrix Type {choice}')
    plt.legend()
    plt.savefig(f'type{choice}_iterations.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_runtime_comparison(nmin, nmax, runtimes_RF_avg, runtimes_SD_avg, runtimes_CG_avg, choice):
    # Use the same step size as the runtime calculations
    dimensions = range(nmin, nmax + 1, 10)

    # Plot runtimes for each method
    plt.plot(dimensions, runtimes_RF_avg, label='RF Runtime', color='g')
    plt.plot(dimensions, runtimes_SD_avg, label='SD Runtime', color='b')
    plt.plot(dimensions, runtimes_CG_avg, label='CG Runtime', color='y')
    plt.xlabel('Dimension n')
    plt.ylabel('Runtime (seconds)')
    plt.title(f'Runtime Comparison for RF, SD, and CG Methods (Type {choice})')
    plt.legend()
    plt.savefig(f'runtime_comparison_{choice}.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_error_ratios(error_ratios, kappa_list, method, nmin, choice):
    # Error ratio plots for RF, SD, or CG
    fig, axes = plt.subplots(5, 2, figsize=(15, 20))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i >= len(error_ratios):  # Avoid IndexError
            break
        ax.plot(range(len(error_ratios[i])), error_ratios[i], label="Error Ratio")
        if method == 'CG':
            bound = (np.sqrt(kappa_list[i]) - 1) / (np.sqrt(kappa_list[i]) + 1)
        else:
            bound = (kappa_list[i] - 1) / (kappa_list[i] + 1)
        ax.axhline(bound, color='r', linestyle='--', label="Convergence Bound")
        ax.set_title(f"{method}: Dimension = {(nmin + i * 10)}")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Error Ratio")
        ax.legend()
    plt.tight_layout()
    plt.savefig(f"{method}_error_ratio_{choice}.png", dpi=300)
    plt.show()

"""
-------------------- Main Routine --------------------
"""
def main():
    nmin, nmax, xmin, xmax, lmin, lmax, choice = get_user_inputs()

    ndim = []
    RF_iter_avg = []
    SD_iter_avg = []
    CG_iter_avg = []
    runtimes_RF_avg = []
    runtimes_SD_avg = []
    runtimes_CG_avg = []
    RF_err_ratio = []
    SD_err_ratio = []
    CG_err_ratio = []
    kappa_list = []

    # Iterating through dimensions
    for n in range(nmin, nmax + 1, 10):
        RF_iter = []
        SD_iter = []
        CG_iter = []
        runtimes_RF = []
        runtimes_SD = []
        runtimes_CG = []
        ndim.append(n)

        # Generate random inputs
        for _ in range(3):
            x_tilde = np.random.uniform(xmin, xmax, n)
            x0 = np.random.uniform(xmin, xmax, n)
            A = part_1_driver(choice, n, lmin, lmax)
            b_tilde = A * x_tilde
            kappa = np.max(A) / np.min(A)
            kappa_list.append(kappa)

            # Measure runtimes and solve
            start = time.time()
            res_RF = richardsons_stationary(A, x_tilde, x0, b_tilde)
            runtimes_RF.append(time.time() - start)

            start = time.time()
            res_SD = steepest_descent(A, x_tilde, x0, b_tilde)
            runtimes_SD.append(time.time() - start)

            start = time.time()
            res_CG = conjugate_gradient(A, x_tilde, x0, b_tilde)
            runtimes_CG.append(time.time() - start)

            # Append iteration counts and error ratios
            RF_iter.append(res_RF[1])
            SD_iter.append(res_SD[1])
            CG_iter.append(res_CG[1])
            RF_err_ratio.append(res_RF[4])
            SD_err_ratio.append(res_SD[4])
            CG_err_ratio.append(res_CG[4])

        # Compute averages
        RF_iter_avg.append(np.mean(RF_iter))
        SD_iter_avg.append(np.mean(SD_iter))
        CG_iter_avg.append(np.mean(CG_iter))
        runtimes_RF_avg.append(np.mean(runtimes_RF))
        runtimes_SD_avg.append(np.mean(runtimes_SD))
        runtimes_CG_avg.append(np.mean(runtimes_CG))

    # Generate plots
    plot_convergence(ndim, RF_iter_avg, SD_iter_avg, CG_iter_avg, choice)
    plot_runtime_comparison(nmin, nmax, runtimes_RF_avg, runtimes_SD_avg, runtimes_CG_avg, choice)
    plot_error_ratios(RF_err_ratio, kappa_list, 'RF', nmin, choice)
    plot_error_ratios(SD_err_ratio, kappa_list, 'SD', nmin, choice)
    plot_error_ratios(CG_err_ratio, kappa_list, 'CG', nmin, choice)


if __name__ == "__main__":
    main()









