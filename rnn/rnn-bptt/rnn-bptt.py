import numpy as np

def bptt_single_step(dh_next: np.ndarray, h_t: np.ndarray, h_prev: np.ndarray,
                     x_t: np.ndarray, W_hh: np.ndarray) -> tuple:
    """
    Backprop through one RNN time step.
    Returns (dh_prev, dW_hh).
    """
    # 1. Compute the pre-activation gradient (local derivative of tanh)
    # dtanh = (1 - h_t^2) * dh_next (element-wise multiplication)
    dtanh = (1 - h_t ** 2) * dh_next
    
    # 2. Compute the gradient with respect to the hidden-to-hidden weights
    # dW_hh = dtanh.T * h_prev
    dW_hh = np.dot(dtanh.T, h_prev)
    
    # 3. Compute the gradient flowing to the previous hidden state
    # dh_prev = dtanh * W_hh
    dh_prev = np.dot(dtanh, W_hh)
    
    return dh_prev, dW_hh