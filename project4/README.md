This report demonstrates and compares the capabilities of the Regula Falsi (false position), Secant, Newton's (Newton-Raphson), and Steffensen's methods for approximating the roots (denoted $x^*$) of real-valued functions. Through a series of controlled experiments, we investigate the convergence behaviors for each root-finding method.

In particular, our first set of tests focuses on higher-order root problems of the form:
\
$f(x) = (x - \rho)^d$,
\
where we observe convergence rate trends in response to changes in the multiplicity ($d$) of the root ($x^* = \rho$).

We then explore the performance of Newton's method on a generic cubic polynomial with three distinct roots ($x^* = 0$, $\rho$, $-\rho$):
\
$f(x) = x^3 - \rho^2 x, \quad \rho > 0$,
\
and investigate Newton's sensitivity to the scaling of $f(x)$ such that:
\
$\tilde{f}(x) = \sigma f(x), \quad \sigma > 0$.
\

The last set of experiments considers the gradual degradation of convergence rates from quadratic to linear to slower linear. We consider the function:
\
$f(x) = x(x - \rho_1)(x - \rho_2), \quad \rho_1 > 0, \rho_2 > 0$,
\
and vary $\rho_1$ and $\rho_2$ to examine how parameter changes affect the convergence rate behaviors of Newton's and Secant methods.

Ultimately, we explore the influence of initial guesses and parameter choices on convergence or divergence for each method. After carefully selecting a small number of illustrative examples and analyzing their outcomes, we discuss when and how each method achieves its expected rate of convergence and under what conditions the performance of each method may degrade.
