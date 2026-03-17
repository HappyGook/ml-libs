import numpy as np

a = np.array([[1,1,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])

print(f"eigenvalues: {np.linalg.eig(a)[0]}")
print(f"eigenvectors: \n {np.linalg.eig(a)[1]}")