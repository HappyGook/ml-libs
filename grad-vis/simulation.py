import numpy as np
import optimizers as opt
import functions as f
import render as r

# Run of the project

configs = {
    1: {"f": f.bowl, "grad": f.grad_bowl},
    2: {"f": f.valley, "grad": f.grad_valley},
    3: {"f": f.rosenbrock, "grad": f.grad_rb},
    4: {"f": f.saddle, "grad": f.grad_saddle},
}
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
x0 = np.array([
    float(input("Initial position (X): ")),
    float(input("Initial position (Y): "))
])

cfg = configs[choice]
gradient = cfg["grad"]
func = cfg["f"]

classic_path = opt.classic_descent(
    x0,
    step_size=step_size,
    iters=iterations,
    gradient=gradient
)

momentum_path = opt.momentum_descent(
        x0,
        step_size=step_size,
        iters=iterations,
        gradient=gradient,
        alpha=alpha
    )

print(f"Computed paths: "
      f"\n Classic descent: {classic_path}"
      f"\n Momentum descent: {momentum_path}")

classic_path = np.load("classic.npy")
momentum_path = np.load("momentum.npy")