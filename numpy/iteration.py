import numpy as np

np1 = np.array([[[1,2,3],[4,5,6],[7,8,9]],[[1,2,3],[4,5,6],[7,8,9]],[[1,2,3],[4,5,6],[7,8,9]]])
print(np1)
for i in np.nditer(np1): # instead of 3 inlay loops np.nditer
    print(i)