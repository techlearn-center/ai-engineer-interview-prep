"""
SOLUTIONS - Logistic Regression from Scratch
===============================================
Try to solve the problems yourself first!
"""
import numpy as np


class LogisticRegression:
    """
    Key concepts to explain in interview:
    - Logistic regression = linear regression + sigmoid activation
    - Sigmoid squishes any value to (0, 1) range -> probability
    - We use binary cross-entropy loss instead of MSE
    - The gradient has the same form as linear regression (convenient!)
    """

    def __init__(self):
        self.weights = None
        self.bias = None

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Key insight: Clip z to [-500, 500] to prevent overflow in exp().
        """
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 1000):
        n_samples, n_features = X.shape

        # Initialize
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(epochs):
            # Forward pass
            z = X @ self.weights + self.bias
            y_pred = self.sigmoid(z)

            # Gradients (same form as linear regression!)
            error = y_pred - y
            dw = (1 / n_samples) * (X.T @ error)
            db = (1 / n_samples) * np.sum(error)

            # Update
            self.weights -= lr * dw
            self.bias -= lr * db

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.weights + self.bias
        return self.sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(y_true == y_pred)
