import numpy as np

#8. 2D Array (Border 1, Inside 0)
#Write a NumPy program to create a 2D array with 1 on the border and 0 inside.
arr = np.full((5,5), 1)
arr[1:-1,1:-1].fill(0)
print(f"8\n{arr}\n")

# 9. Add Border to Array (0s)
# Write a NumPy program to add a border (filled with 0's)
# around an existing array.
x = 3
borderless = np.full((x,x),1)
bordered = np.full((x+2,x+2),0)
bordered[1:-1,1:-1] = borderless
print(f"bordered array: \n{bordered}\n")

# 10. 8x8 Checkerboard Pattern
# Write a NumPy program to create an 8x8 matrix and
# fill it with a checkerboard pattern.
checkerboard = np.full((8,8),1)
checkerboard[:,0:-1:2].fill(0)
print(f"10\n{checkerboard}\n")

# 11 --?

# 12
b = np.matrix([[1,1,1],[1,2,0],[1,1,0]])
b_inv = np.linalg.inv(b)
a = np.matrix([[1,1,0],[1,-1,0],[1,1,1]])
print(b)
print(b_inv)
print(b_inv*a*b)

matr=np.array([[5,-2,1],[-2,1,0],[1,0,1]])
print(np.linalg.eigvals(matr))