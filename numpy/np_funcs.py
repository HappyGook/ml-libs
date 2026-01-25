import numpy as np

# Universal funcs in Numpy https://numpy.org/doc/stable/reference/ufuncs.html
np1 = np.array([4,16,36,64,128,333,4,1])
np2 = np.array([65,6,12,235,89,2,8,3])
neg = np.array([-4,-16,-36,64,-128,333,4,0])

# transforms each element into its square root
print(np.sqrt(np1))

# transforms each element into it's abs value
print(np.absolute(neg))

# transforms each element into exponentials
print(np.exp(np1))

# finds min/max from array
print(f"max: {np.max(np1)}, min: {np.min(np1)}")

# transforms negative elements into -1, 0 as 0, positive as 1
print(np.sign(neg))

# Trigonometrical funcs sin,cos, tan or also
# sinh, tanh, cosh - hyperbolic
# arcsin, arccos, arctan
print(np.tanh(np1))

# logarithmic log / log2 / log10 / logn via np.emath
print(np.log(np1))

# Greatest Common Divisor / Lowest Common Multiple
print(np.gcd(np1,neg))
print(np.lcm(np1,neg))

#Also there are bitwise operations, comparisons, logical funcs, float funcs


# Scalar arithmetics

print(np1+1)
print(np1 - 1)
print(np1*2)
print(np1/5)
print(np1**(1/2))

# Vectorized functions
# sqrt, round, floor/ceil, log, abs, pi, etc.
print(np.floor(np.sqrt(np1)))

# In two arrays, operations are applied element-wise
print(np1+np2)
print(np1-np2)
print(np1*np2) # not a matrix multiplication
print(np1/np2)

# Matrix multiplication example (Works the same with np.matrix objects)
matrix1 = np.array([[1,2,3],[3,4,5],[4,5,6]])
matrix2 = np.array([[564,12,6],[12,5,6],[9,76,6]])

# Possibilities are @ operator, np.matmul, np.dot <- counts products of arrays
print(f"Matrices multiplied: \n{matrix1@matrix2}\n")
print(f"Matrices multiplied: \n{np.matmul(matrix1,matrix2)}\n")

# Matrix division ( A * B^-1 )
print(f"Matrices divided: \n{matrix1@(np.linalg.inv(matrix2))}\n")