import numpy as np

# Sorting the arrays
np1 = np.array([2,7,35,768,32,68])
print(np.sort(np1))

np_str = np.array(["word","second_word","first_word"])
print(np.sort(np_str))

# For multi-dim arrays, the sorting sorts inside
np_bool = np.array([[True, False, True],[True, True, False],[False, False, True]])
print(np.sort(np_bool)) # Normal sort sorts inside individual rows (axis=-1)
print(np.sort(np_bool, axis=None)) # axis=None flattens before sorting
print(np.sort(np_bool, axis=1)) # Sort along rows (row-index)
print(np.sort(np_bool, axis=0)) # Sort among columns (column-index)

np2 = np.random.randint(10, size=(3,3))
print(f"\noriginal array: \n{np2}\n")
print(f"Sorted with axis=-1: \n{np.sort(np2)}")
print(f"Sorted with axis=None: \n{np.sort(np2,axis=None)}")
print(f"Sorted with axis=1: \n{np.sort(np2, axis=1)}")
print(f"Sorted with axis=0: \n{np.sort(np2, axis=0)}")
"""
original array: 
[[5 9 6]
 [9 8 6]
 [9 6 2]]
 
Axis=0 sorting:
Column 0: [5, 9, 9]
Column 1: [9, 8, 6]
Column 2: [6, 6, 2]

Each column is sorted individually
Column 0 → [5, 9, 9]
Column 1 → [6, 8, 9]
Column 2 → [2, 6, 6]

Values are returned row-by-row
Row 0: [5, 6, 2]
Row 1: [9, 8, 6]
Row 2: [9, 9, 6]
"""

# if descending sorting is needed => reverse results
print(f"Original np2 again: \n{np2}")
print(f"\n Reversed sorting of np2 (Along Row): \n{np.sort(np2)[:,::-1]}")
print(f"Reversed sorting of np2 (Along Column): \n{np.sort(np2)[::-1,:]}")
print(f"\n Other option for reversed sorting of np2 (Along Rows): \n{np.sort(-np2,axis=1) * -1}")

# For further sorting options => argsort