import numpy as np

class VanillaRNN:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        
        # Xavier initialization
        self.W_xh = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / (2 * hidden_dim))
        self.W_hy = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_h = np.zeros(hidden_dim)
        self.b_y = np.zeros(output_dim)
        
    def forward(self, X: np.ndarray, h_0: np.ndarray = None) -> tuple:
        """
        Forward pass through entire sequence.
        Returns (y_seq, h_final).
        """
        # Extract dimensions
        batch_size, T, input_dim = X.shape
        output_dim = self.W_hy.shape[0]
        
        # Initialize hidden state to zeros if not provided
        if h_0 is None:
            h_t = np.zeros((batch_size, self.hidden_dim))
        else:
            h_t = h_0
            
        # Array to store the output logits for each time step
        y_seq = np.zeros((batch_size, T, output_dim))
        
        # Iterate over the sequence length
        for t in range(T):
            x_t = X[:, t, :]
            
            # Recurrent computation for the current hidden state
            # h_t = tanh(x_t * W_xh^T + h_{t-1} * W_hh^T + b_h)
            h_t = np.tanh(np.dot(x_t, self.W_xh.T) + np.dot(h_t, self.W_hh.T) + self.b_h)
            
            # Output projection for the current time step
            # y_t = h_t * W_hy^T + b_y
            y_t = np.dot(h_t, self.W_hy.T) + self.b_y
            
            # Store the output
            y_seq[:, t, :] = y_t
            
        h_final = h_t
        
        return y_seq, h_final