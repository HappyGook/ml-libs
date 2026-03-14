import numpy as np
import optimizers as opt
import functions as f

# Run of the project

choice = 0
while 1<=choice<=4:
    choice = int(input("Choose which surface function do you want to compute a descent for:\n"
                   "1 - convex bowl (f(x,y) = x^2 + y^2)\n"
                   "2 - elliptical valley  (f(x,y) = x^2 - 10*y^2)\n"
                   "3 - Rosenbrock valley (f(x,y) = (1 - x)^2 + 100(y-x^2)^2)\n"
                   "4 - Saddle (f(x,y) = x^2 - y^2)\n"))

step_size = float(input("Step size: "))
iterations = int(input("Number of iterations:"))
alpha = float(input("Choose an alpha for momentum descent: "))
x0 = np.array([2])
x0[0] = float(input("Initial position (X):"))
x0[1] = float(input("Initial position (Y):"))

if choice == 1:
    classic_path = opt.classic_descent(
        x0,
        step_size=step_size,
        iters=iterations,
        gradient=f.grad_bowl
    )
    momentum_path = opt.momentum_descent(
        x0,
        step_size=step_size,
        iters=iterations,
        gradient=f.grad_bowl,
        alpha=alpha
    )
elif choice == 2:
    classic_path = opt.classic_descent(
        x0,
        step_size=step_size,
        iters=iterations,
        gradient=f.grad_valley
    )
    momentum_path = opt.momentum_descent(
        x0,
        step_size=step_size,
        iters=iterations,
        gradient=f.grad_valley,
        alpha=alpha
    )
elif choice == 3:
    classic_path = opt.classic_descent(
        x0,
        step_size=step_size,
        iters=iterations,
        gradient=f.grad_rb
    )
    momentum_path = opt.momentum_descent(
        x0,
        step_size=step_size,
        iters=iterations,
        gradient=f.grad_rb,
        alpha=alpha
    )
elif choice == 4:
    classic_path = opt.classic_descent(
        x0,
        step_size=step_size,
        iters=iterations,
        gradient=f.grad_saddle
    )
    momentum_path = opt.momentum_descent(
        x0,
        step_size=step_size,
        iters=iterations,
        gradient=f.grad_saddle,
        alpha=alpha
    )

