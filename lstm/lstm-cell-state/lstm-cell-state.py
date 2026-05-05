import numpy as np

def update_cell_state(C_prev: np.ndarray, f_t: np.ndarray, 
                      i_t: np.ndarray, c_tilde: np.ndarray) -> np.ndarray:
    """Update cell state: C_t = f_t * C_prev + i_t * c_tilde"""
    
    # 1. Element-wise multiply the forget gate output (f_t) 
    #    with the previous cell state (C_prev). This discards old info.
    forget_part = f_t * C_prev
    
    # 2. Element-wise multiply the input gate output (i_t) 
    #    with the candidate memory (c_tilde). This scales the new info.
    input_part = i_t * c_tilde
    
    # 3. Add the two parts together to get the new cell state
    C_t = forget_part + input_part
    
    return C_t