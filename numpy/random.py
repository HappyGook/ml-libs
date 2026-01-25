import numpy as np

# rng object used to generate random values
# seed makes the random return an exact pack of values
rng = np.random.default_rng(seed=42)

print(rng.integers(1,10, size=(3,2)))

# uniform returns random floats with the same chance for each float
# empty parameters for range [0,1)
print(np.random.uniform(1,2))

np1 = np.array([6,3,56,2,5,4])
# Shuffles an array randomly / to given seed
rng.shuffle(np1)
print(np1)

# Random choice can be done via object or np method
chosen_array = rng.choice(np1, size=(2,2))
chosen_array2 = np.random.choice(np1,size=(2,2))

print(f"\n {chosen_array}\n {chosen_array2}")
