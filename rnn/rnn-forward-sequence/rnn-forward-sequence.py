import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray, 
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    # Extract dimensions
    batch_size, T, input_dim = X.shape
    hidden_dim = h_0.shape[1]
    
    # Initialize an array to store the hidden states for all time steps
    hidden_states = np.zeros((batch_size, T, hidden_dim))
    
    # Set the initial hidden state
    h_t = h_0
    
    # Loop over each time step in the sequence
    for t in range(T):
        # Extract the input at the current time step (shape: batch_size, input_dim)
        x_t = X[:, t, :]
        
        # Apply the Vanilla RNN formula:
        # h_t = tanh(x_t * W_xh^T + h_{t-1} * W_hh^T + b_h)
        h_t = np.tanh(np.dot(x_t, W_xh.T) + np.dot(h_t, W_hh.T) + b_h)
        
        # Store the current hidden state in our sequence array
        hidden_states[:, t, :] = h_t
        
    # The last hidden state is returned separately as h_final
    h_final = h_t
    
    return hidden_states, h_final