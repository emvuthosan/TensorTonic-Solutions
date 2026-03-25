import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L)
    """
    N = len(seqs)
    
    if max_len is None:
        max_len = max(len(seq) for seq in seqs) if seqs else 0
    
    result = np.full((N, max_len), pad_value)
    
    for i, seq in enumerate(seqs):
        length = min(len(seq), max_len)
        result[i, :length] = seq[:length] 
    
    return result

