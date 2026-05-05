import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    """
    Simulate gradient norm decay over T time steps.
    Returns list of gradient norms.
    """
    # Calculate the spectral norm (L2 norm) of the weight matrix W_hh.
    spectral_norm = np.linalg.norm(W_hh, ord=2)
    
    gradient_norms = []
    current_norm = 1.0
    
    # Propagate backward through T time steps
    for _ in range(T):
        # 1. Record the gradient norm at the current step FIRST
        gradient_norms.append(float(current_norm))
        
        # 2. Decay/grow the norm for the previous time step
        current_norm *= spectral_norm
        
    return gradient_norms