"""
SOLUTIONS - Feature Engineering
=================================
Try to solve the problems yourself first!
"""
import numpy as np


def one_hot_encode(labels: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    Key insight: Sort unique labels, create a mapping, then build the matrix.
    """
    unique_labels = sorted(set(labels))
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}

    n_samples = len(labels)
    n_classes = len(unique_labels)
    encoded = np.zeros((n_samples, n_classes), dtype=int)

    for i, label in enumerate(labels):
        encoded[i, label_to_idx[label]] = 1

    return encoded, unique_labels


def min_max_scale(X: np.ndarray) -> np.ndarray:
    """
    Key insight: Compute min/max per column using axis=0.
    Handle the edge case where max == min.
    """
    X = X.astype(float)
    col_min = X.min(axis=0)
    col_max = X.max(axis=0)
    range_vals = col_max - col_min

    # Avoid division by zero
    range_vals[range_vals == 0] = 1  # will result in 0 since numerator is also 0

    return (X - col_min) / range_vals


def add_polynomial_features(X: np.ndarray, degree: int = 2) -> np.ndarray:
    """
    Key insight: Stack X^1, X^2, ..., X^degree as columns.
    np.hstack or np.column_stack works well here.
    """
    features = []
    for d in range(1, degree + 1):
        features.append(X ** d)
    return np.hstack(features)


def handle_missing_values(X: np.ndarray, strategy: str = "mean") -> np.ndarray:
    """
    Key insight: np.nanmean and np.nanmedian ignore NaN values.
    Use np.isnan to find and replace NaN values column by column.
    """
    X = X.copy().astype(float)
    n_cols = X.shape[1]

    for col in range(n_cols):
        mask = np.isnan(X[:, col])
        if not mask.any():
            continue

        if strategy == "mean":
            fill_value = np.nanmean(X[:, col])
        elif strategy == "median":
            fill_value = np.nanmedian(X[:, col])
        elif strategy == "zero":
            fill_value = 0
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        X[mask, col] = fill_value

    return X
