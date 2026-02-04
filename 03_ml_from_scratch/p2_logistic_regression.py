"""
Problem 2: Logistic Regression from Scratch
=============================================
Difficulty: Medium

Implement binary logistic regression using only NumPy.

Run tests:
    pytest 03_ml_from_scratch/tests/test_p2_logistic_regression.py -v
"""
import numpy as np


class LogisticRegression:
    """
    Binary logistic regression with gradient descent.
    """

    def __init__(self):
        self.weights = None
        self.bias = None

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation: 1 / (1 + exp(-z))
        Handle numerical stability (clip z to avoid overflow).
        """
        # YOUR CODE HERE
        pass

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 1000):
        """
        Train using gradient descent on binary cross-entropy loss.

        X: shape (n_samples, n_features)
        y: shape (n_samples,) with values 0 or 1

        Gradient update:
            z = X @ weights + bias
            y_pred = sigmoid(z)
            dw = (1/n) * X.T @ (y_pred - y)
            db = (1/n) * sum(y_pred - y)
            weights -= lr * dw
            bias -= lr * db
        """
        # YOUR CODE HERE
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of class 1."""
        # YOUR CODE HERE
        pass

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return class labels (0 or 1) based on threshold."""
        # YOUR CODE HERE
        pass

    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute classification accuracy."""
        # YOUR CODE HERE
        pass
