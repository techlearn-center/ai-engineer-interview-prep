"""
SOLUTIONS - Linear Regression from Scratch
=============================================
Try to solve the problems yourself first!
"""
import numpy as np


class LinearRegression:
    """
    Key concepts to explain in interview:
    - We're minimizing MSE loss using gradient descent
    - The gradient tells us the direction of steepest increase
    - We move OPPOSITE to the gradient (hence the minus sign)
    - Learning rate controls step size
    """

    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000):
        n_samples, n_features = X.shape

        # Initialize weights to zeros
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # Gradient descent loop
        for _ in range(epochs):
            # Forward pass: compute predictions
            y_pred = X @ self.weights + self.bias

            # Compute gradients
            error = y_pred - y
            dw = (2 / n_samples) * (X.T @ error)
            db = (2 / n_samples) * np.sum(error)

            # Update parameters (move opposite to gradient)
            self.weights -= lr * dw
            self.bias -= lr * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)
