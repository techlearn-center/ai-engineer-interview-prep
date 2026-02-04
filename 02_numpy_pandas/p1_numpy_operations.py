"""
Problem 1: NumPy Array Operations
===================================
Difficulty: Easy -> Medium

NumPy is the backbone of ML in Python. You WILL be tested on this.

Run tests:
    pytest 02_numpy_pandas/tests/test_p1_numpy_operations.py -v
"""
import numpy as np


def normalize_array(arr: np.ndarray) -> np.ndarray:
    """
    Normalize an array to have mean=0 and std=1 (Z-score normalization).
    This is one of the most common preprocessing steps in ML.

    Formula: (x - mean) / std

    Example:
        normalize_array(np.array([1, 2, 3, 4, 5]))
        -> array([-1.414, -0.707, 0.0, 0.707, 1.414])  (approximately)
    """
    # YOUR CODE HERE
    pass


def batch_dot_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Given two 2D arrays of shape (n, d), compute the dot product of
    each corresponding row pair. Return a 1D array of shape (n,).

    Do NOT use a loop. Use vectorized operations.

    Example:
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        -> array([17, 53])  # (1*5+2*6, 3*7+4*8)
    """
    # YOUR CODE HERE
    pass


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise cosine similarity matrix for rows in X.
    X has shape (n, d). Return shape (n, n).

    cosine_sim(a, b) = (a . b) / (||a|| * ||b||)

    This is CRITICAL for embeddings, search, and RAG systems.
    Do NOT use loops - use matrix operations.

    Example:
        X = np.array([[1, 0], [0, 1], [1, 1]])
        -> 3x3 matrix of pairwise cosine similarities
    """
    # YOUR CODE HERE
    pass


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Implement the softmax function for a 1D array.
    softmax(x_i) = exp(x_i) / sum(exp(x_j))

    For numerical stability, subtract max(x) before exponentiating.
    This is fundamental to classification models and attention mechanisms.

    Example:
        softmax(np.array([1, 2, 3]))
        -> array([0.0900, 0.2447, 0.6652])  (approximately)
    """
    # YOUR CODE HERE
    pass


def broadcast_operation(prices: np.ndarray, discounts: np.ndarray) -> np.ndarray:
    """
    Given:
        prices: shape (n_products,) - price of each product
        discounts: shape (n_tiers,) - discount percentage for each tier

    Return a 2D array of shape (n_products, n_tiers) where element [i, j]
    is the discounted price of product i at tier j discount.

    discounted_price = price * (1 - discount/100)

    Use NumPy broadcasting (no loops).

    Example:
        prices = np.array([100, 200, 300])
        discounts = np.array([10, 20])
        -> array([[90, 80], [180, 160], [270, 240]])
    """
    # YOUR CODE HERE
    pass
