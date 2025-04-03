import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return (x**3) - (2 * (x**2)) + 10
def f2(x):
    return ((1/4) * (x**4)) - ((2/3) * (x**3)) + (10 * x)
x = np.linspace(0, 3, 400)
y = f(x)
y2 = f2(x)
ymax1 = f2(1)
ymax2 = f2(2)
plt.figure()
plt.plot(x, y, label=r'$f(x) = x^3 - 2x^2 + 10$')
plt.plot(x, y2, label=r'$F(x) = \frac{1}{4}x^4 - \frac{2}{3}x^3 + 10x$')
plt.plot([1, 1], [0, ymax1], color='r', linestyle='--')
plt.plot([2, 2], [0, ymax2], color='r', linestyle='--', label='interval of integration [1,2]')
plt.fill_between(x, 0, y, where=((x >= 1) & (x <= 2)), color='skyblue', alpha=0.4)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fundamental Theorem of Calculus')
plt.grid(True)
plt.legend()
plt.savefig('ftoc.png')
plt.show()
