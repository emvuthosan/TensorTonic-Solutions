import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def lstm_cell(x_t: np.ndarray, h_prev: np.ndarray, C_prev: np.ndarray,
              W_f: np.ndarray, W_i: np.ndarray, W_c: np.ndarray, W_o: np.ndarray,
              b_f: np.ndarray, b_i: np.ndarray, b_c: np.ndarray, b_o: np.ndarray) -> tuple:
    """Complete LSTM cell forward pass."""
    
    # 1. Concatenate previous hidden state and current input
    concat = np.concatenate([h_prev, x_t], axis=-1)
    
    # 2. Compute Forget Gate (f_t)
    f_t = sigmoid(np.dot(concat, W_f.T) + b_f)
    
    # 3. Compute Input Gate (i_t) and Candidate Memory (c_tilde)
    i_t = sigmoid(np.dot(concat, W_i.T) + b_i)
    c_tilde = np.tanh(np.dot(concat, W_c.T) + b_c)
    
    # 4. Compute Output Gate (o_t)
    o_t = sigmoid(np.dot(concat, W_o.T) + b_o)
    
    # 5. Update Cell State (C_t)
    C_t = f_t * C_prev + i_t * c_tilde
    
    # 6. Compute Hidden State (h_t)
    h_t = o_t * np.tanh(C_t)
    
    # 7. Return new hidden state and cell state as a tuple
    return h_t, C_t