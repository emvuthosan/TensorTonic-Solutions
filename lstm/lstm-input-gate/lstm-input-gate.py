import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def input_gate(h_prev: np.ndarray, x_t: np.ndarray,
               W_i: np.ndarray, b_i: np.ndarray,
               W_c: np.ndarray, b_c: np.ndarray) -> tuple:
    """Compute input gate and candidate memory."""
    
    # 1. Concatenate h_prev and x_t along the last axis
    concat = np.concatenate([h_prev, x_t], axis=-1)
    
    # 2. Compute the Input Gate (i_t)
    # Applies linear transformation followed by sigmoid activation
    linear_i = np.dot(concat, W_i.T) + b_i
    i_t = sigmoid(linear_i)
    
    # 3. Compute the Candidate Memory (c_tilde)
    # Applies linear transformation followed by tanh activation
    linear_c = np.dot(concat, W_c.T) + b_c
    c_tilde = np.tanh(linear_c)
    
    # 4. Return both as a tuple
    return i_t, c_tilde