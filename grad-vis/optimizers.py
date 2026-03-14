import numpy as np

# Different gradient descent algorithms

# Simple gradient descent always computes gradients with the same weight
def classic_descent(x0, step_size, iters, gradient):
    xs = [x0]
    x=x0.copy()
    for _ in range(iters):
        x = x - step_size * gradient(x)
        xs.append(x.copy())
    return np.array(xs)

# Gradient with momentum computes updates as a linear combination
def momentum_descent(x0, step_size, iters, gradient, alpha):
    xs = [x0]
    x = x0.copy()
    delta = np.zeros_like(x)
    for _ in range(iters):
        delta = alpha*delta - step_size * gradient(x)
        x = x + delta
        xs.append(x.copy())
    return np.array(xs)
