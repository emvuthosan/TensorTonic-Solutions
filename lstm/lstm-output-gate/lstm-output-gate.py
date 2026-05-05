import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def output_gate(h_prev: np.ndarray, x_t: np.ndarray, C_t: np.ndarray,
                W_o: np.ndarray, b_o: np.ndarray) -> tuple:
    """Compute output gate and hidden state."""
    
    # 1. Concatenate the previous hidden state and current input
    concat = np.concatenate([h_prev, x_t], axis=-1)
    
    # 2. Compute the Output Gate (o_t)
    # Apply the linear transformation followed by the sigmoid activation
    linear_o = np.dot(concat, W_o.T) + b_o
    o_t = sigmoid(linear_o)
    
    # 3. Compute the new Hidden State (h_t)
    # Push the cell state through tanh, then multiply element-wise by the output gate
    h_t = o_t * np.tanh(C_t)
    
    # 4. Return both as a tuple
    return o_t, h_t