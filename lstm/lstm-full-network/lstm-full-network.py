import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class LSTM:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Initialize weights
        self.W_f = np.random.randn(hidden_dim, hidden_dim + input_dim)
        self.W_i = np.random.randn(hidden_dim, hidden_dim + input_dim)
        self.W_c = np.random.randn(hidden_dim, hidden_dim + input_dim)
        self.W_o = np.random.randn(hidden_dim, hidden_dim + input_dim)

        self.b_f = np.zeros(hidden_dim)
        self.b_i = np.zeros(hidden_dim)
        self.b_c = np.zeros(hidden_dim)
        self.b_o = np.zeros(hidden_dim)

        self.W_y = np.random.randn(output_dim, hidden_dim)
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray) -> tuple:
        # X shape: (1, T, input_dim)
        batch_size, T, _ = X.shape

        # initialize hidden + cell
        h = np.zeros((batch_size, self.hidden_dim))
        C = np.zeros((batch_size, self.hidden_dim))

        outputs = []

        for t in range(T):
            x_t = X[:, t, :]  # (1, input_dim)

            combined = np.concatenate([h, x_t], axis=1)  # (1, hidden+input)

            f = sigmoid(combined @ self.W_f.T + self.b_f)
            i = sigmoid(combined @ self.W_i.T + self.b_i)
            c_tilde = np.tanh(combined @ self.W_c.T + self.b_c)
            o = sigmoid(combined @ self.W_o.T + self.b_o)

            C = f * C + i * c_tilde
            h = o * np.tanh(C)

            y_t = h @ self.W_y.T + self.b_y  # (1, output_dim)

            outputs.append(y_t)

        # stack theo time axis
        outputs = np.stack(outputs, axis=1)  # (1, T, output_dim)

        return outputs, h, C