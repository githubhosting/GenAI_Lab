# Question 10 : Implementation of the self-attention mechanism using only NumPy

import numpy as np


def softmax(x, axis=-1):
    """Compute the softmax of each element along the specified axis of x."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True)
                   )  # For numerical stability
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def self_attention(X, Wq, Wk, Wv):
    """
    Implement the self-attention mechanism.

    Args:
        X: Input matrix of shape (n, d), where n is the number of input vectors, and d is the dimension of each vector.
        Wq: Query weight matrix of shape (d, dout).
        Wk: Key weight matrix of shape (d, dout).
        Wv: Value weight matrix of shape (d, dout).

    Returns:
        Output matrix of shape (n, dout).
    """
    # Compute Queries (Q), Keys (K), and Values (V)
    Q = np.dot(X, Wq)  # Shape: (n, dout)
    K = np.dot(X, Wk)  # Shape: (n, dout)
    V = np.dot(X, Wv)  # Shape: (n, dout)

    # Compute attention scores: Q * K.T, then scale by sqrt(dout)
    d_k = Q.shape[1]  # dout
    attention_scores = np.dot(Q, K.T) / np.sqrt(d_k)  # Shape: (n, n)

    # Apply softmax to attention scores
    attention_weights = softmax(attention_scores, axis=-1)  # Shape: (n, n)

    # Compute final output: Attention weights * V
    output = np.dot(attention_weights, V)  # Shape: (n, dout)

    return output


# Example usage:
np.random.seed(0)  # For reproducibility

# Input matrix X (n=4 vectors, d=3 features per vector)
X = np.random.rand(4, 3)  # Shape: (4, 3)

# Learnable weight matrices Wq, Wk, Wv
d = 3      # Input dimension
dout = 2   # Output dimension
Wq = np.random.rand(d, dout)  # Shape: (3, 2)
Wk = np.random.rand(d, dout)  # Shape: (3, 2)
Wv = np.random.rand(d, dout)  # Shape: (3, 2)

# Call the self_attention function
output = self_attention(X, Wq, Wk, Wv)

print("Input Matrix X:")
print(X)
print("\nWeight Matrix Wq:")
print(Wq)
print("\nWeight Matrix Wk:")
print(Wk)
print("\nWeight Matrix Wv:")
print(Wv)
print("\nSelf-Attention Output:")
print(output)
