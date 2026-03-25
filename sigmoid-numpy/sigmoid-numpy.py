import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    x = np.array(x)
    return 1 / (1 + np.exp(-x))

# Test
x = np.array([0, 2, -2])
print(sigmoid(x))