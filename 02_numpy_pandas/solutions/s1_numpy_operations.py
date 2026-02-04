"""
SOLUTIONS - NumPy Operations
==============================
Try to solve the problems yourself first!
"""
import numpy as np


def normalize_array(arr: np.ndarray) -> np.ndarray:
    """
    Key insight: Z-score normalization is (x - mean) / std.
    NumPy vectorizes this - it applies to every element at once.
    """
    mean = arr.mean()
    std = arr.std()
    if std == 0:
        return np.zeros_like(arr)
    return (arr - mean) / std


def batch_dot_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Key insight: Element-wise multiply then sum along axis=1.
    This is equivalent to computing the dot product of each row pair.

    Alternative: np.einsum('ij,ij->i', A, B)
    """
    return np.sum(A * B, axis=1)


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """
    Key insight: cosine_sim(a, b) = (a . b) / (||a|| * ||b||)

    For the whole matrix:
        1. Compute norms: ||x_i|| for each row
        2. Dot product matrix: X @ X.T
        3. Divide by outer product of norms
    """
    # Compute L2 norms for each row
    norms = np.linalg.norm(X, axis=1, keepdims=True)  # shape (n, 1)
    # Normalize rows
    X_normalized = X / norms
    # Dot product of normalized vectors gives cosine similarity
    return X_normalized @ X_normalized.T


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Key insight: Subtract max(x) for numerical stability.
    Without this, exp(1000) would overflow to infinity.
    Subtracting max doesn't change the result mathematically.
    """
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / exp_x.sum()


def broadcast_operation(prices: np.ndarray, discounts: np.ndarray) -> np.ndarray:
    """
    Key insight: Reshape prices to (n, 1) and discounts stays (m,).
    NumPy broadcasting automatically creates the (n, m) result.

    prices[:, None] converts shape (n,) -> (n, 1)
    Then (n, 1) * (m,) -> (n, m) via broadcasting
    """
    return prices[:, None] * (1 - discounts[None, :] / 100)
