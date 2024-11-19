from mypackage import myfunctions
import numpy as np
import matplotlib.pyplot as plt

#Richardon's First Order Stationary Method (RF)
def richardsons_stationary(A, x_tilde, x0, b_tilde):
    # Optimal alpha to guarantee convergence
    alpha = 2 / (np.max(A) + np.min(A))

    # Calculate b = Ax for generated x
    b_tilde = np.dot(A, x_tilde)

    # Initial residual
    r0 = b_tilde - np.dot(A, x0)
    r0_norm = np.linalg.norm(r0)
    r_norm = r0_norm
    resid_arr = [r0_norm]

    # Initial error
    err = np.subtract(x_tilde, x0)
    err_norm = np.linalg.norm(err)
    err_arr = []

    x = x0
    r = r0

    # Termination criterion
    tol = 10**6

    # Number of iterations until convergence
    iter = 0

    # Iterate until norm(current residual) / norm(initial residual) < 10^6
    while((r_norm / r0_norm) > tol):
        x_next = x + (alpha * r) # update x
        r_next = r - np.dot(A, x_next) # update r
        r_norm = np.linalg.norm(r_next)
        resid_arr.append(r_norm)
        err_next = np.subtract(x_next, x) # update error
        step = err_next / err_norm
        err_arr.append(step)

        # Update initial terms for next iteration
        x = x_next
        r = r_next
        err = err_next
        iter += 1

    return x, iter, resid_arr, err_arr

# Steepest Descent (SD)
def steepest_descent(A, x_tilde, x0, b_tilde):



