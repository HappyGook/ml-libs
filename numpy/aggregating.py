import numpy as np

np1 = np.array([[8,3,25],[10,3,1345]])

# Aggregation funcs like sum, mean, std, var and so on
print(np1.sum())
print(np1.mean())
print(np1.std())
print(np1.var())

# Axis works like in sorting
print(np1.sum(axis=0)) # Aggregates column-index
# Returns [  18    6 1370]

# min and max return the elements themselves
print(np1.min()) # 3

# argmin and argmax return the index of the element
print(np1.argmax()) # 5