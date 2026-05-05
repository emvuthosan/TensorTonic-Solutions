import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def forget_gate(h_prev: np.ndarray, x_t: np.ndarray, W_f: np.ndarray, b_f: np.ndarray) -> np.ndarray:
    """Compute forget gate: f_t = sigmoid(W_f @ [h, x] + b_f)"""
    # 1. Concatenate h_prev and x_t along the last axis
    # If inputs are vectors (size D), result is size (H+D)
    # If inputs are batches (N, D), result is (N, H+D)
    concat = np.concatenate([h_prev, x_t], axis=-1)
    
    # 2. Apply linear transformation: weight matrix multiplication and bias addition
    linear_transform = np.dot(concat, W_f.T) + b_f
    
    # 3. Apply sigmoid activation
    f_t = sigmoid(linear_transform)
    
    return f_t