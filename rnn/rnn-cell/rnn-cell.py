import numpy as np

def rnn_cell(x_t: np.ndarray, h_prev: np.ndarray,
             W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Single RNN cell forward pass.
    """

    # reshape nếu cần
    if x_t.ndim == 1:
        x_t = x_t.reshape(1, -1)
    if h_prev.ndim == 1:
        h_prev = h_prev.reshape(1, -1)

    h_t = np.tanh(
        x_t @ W_xh.T +
        h_prev @ W_hh.T +
        b_h
    )

    return h_t.flatten()   