"""
Problem 1: Linear Regression from Scratch
==========================================
Difficulty: Medium

Implement linear regression using only NumPy.
This is the #1 most asked ML coding question in interviews.

Run tests:
    pytest 03_ml_from_scratch/tests/test_p1_linear_regression.py -v
"""
import numpy as np


class LinearRegression:
    """
    Implement simple linear regression with gradient descent.

    Methods to implement:
        - fit(X, y, lr, epochs): Train the model
        - predict(X): Make predictions
        - mse(y_true, y_pred): Compute mean squared error
    """

    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000):
        """
        Train the model using gradient descent.

        X: shape (n_samples, n_features)
        y: shape (n_samples,)
        lr: learning rate
        epochs: number of iterations

        Gradient update rules:
            y_pred = X @ weights + bias
            dw = (2/n) * X.T @ (y_pred - y)
            db = (2/n) * sum(y_pred - y)
            weights -= lr * dw
            bias -= lr * db
        """
        # YOUR CODE HERE
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using learned weights and bias.
        y_pred = X @ weights + bias
        """
        # YOUR CODE HERE
        pass

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Mean Squared Error.
        MSE = (1/n) * sum((y_true - y_pred)^2)
        """
        # YOUR CODE HERE
        pass
