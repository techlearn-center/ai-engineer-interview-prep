"""
Problem 2: Feature Engineering
================================
Difficulty: Medium

Feature engineering is critical for real-world ML. These patterns come up
in interviews when they ask "how would you prepare this data for a model?"

Run tests:
    pytest 04_data_processing/tests/test_p2_feature_engineering.py -v
"""
import numpy as np


def one_hot_encode(labels: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    One-hot encode a list of categorical labels.
    Return (encoded_matrix, unique_labels_sorted).

    Example:
        one_hot_encode(["cat", "dog", "cat", "bird"])
        -> (array([[0, 1, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [1, 0, 0]]),
            ["bird", "cat", "dog"])
    """
    # YOUR CODE HERE
    pass


def min_max_scale(X: np.ndarray) -> np.ndarray:
    """
    Scale features to [0, 1] range column-wise.
    Formula: (x - min) / (max - min) for each column.

    X has shape (n_samples, n_features).
    Handle case where max == min (return 0 for that column).
    """
    # YOUR CODE HERE
    pass


def add_polynomial_features(X: np.ndarray, degree: int = 2) -> np.ndarray:
    """
    Add polynomial features up to the given degree for a SINGLE feature column.
    X has shape (n_samples, 1).

    For degree=3: return [X, X^2, X^3]

    Example:
        X = np.array([[2], [3]])
        add_polynomial_features(X, degree=3)
        -> array([[2, 4, 8], [3, 9, 27]])
    """
    # YOUR CODE HERE
    pass


def handle_missing_values(X: np.ndarray, strategy: str = "mean") -> np.ndarray:
    """
    Replace NaN values in each column using the given strategy.
    Strategies: "mean", "median", "zero"

    X has shape (n_samples, n_features), may contain np.nan.
    Use np.nanmean / np.nanmedian to compute stats ignoring NaN.
    """
    # YOUR CODE HERE
    pass
