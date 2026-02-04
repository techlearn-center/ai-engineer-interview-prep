"""
Problem 3: K-Means Clustering from Scratch
============================================
Difficulty: Medium

Implement K-Means clustering using only NumPy.

Run tests:
    pytest 03_ml_from_scratch/tests/test_p3_kmeans.py -v
"""
import numpy as np


class KMeans:
    """
    K-Means clustering algorithm.
    """

    def __init__(self, k: int = 3, max_iters: int = 100, random_state: int = 42):
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    def fit(self, X: np.ndarray):
        """
        Run K-Means on data X of shape (n_samples, n_features).

        Algorithm:
            1. Randomly initialize k centroids from the data points
            2. Repeat until convergence or max_iters:
                a. Assign each point to nearest centroid
                b. Update centroids to mean of assigned points
            3. Store self.centroids and self.labels
        """
        # YOUR CODE HERE
        pass

    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        """
        Assign each sample to the nearest centroid.
        Return array of labels with shape (n_samples,).

        Hint: compute distances from each point to each centroid using
        vectorized operations (no loops over samples).
        """
        # YOUR CODE HERE
        pass

    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Compute new centroids as the mean of assigned samples.
        Return new centroids of shape (k, n_features).
        """
        # YOUR CODE HERE
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign clusters to new data points."""
        # YOUR CODE HERE
        pass

    @staticmethod
    def inertia(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        """
        Compute inertia (within-cluster sum of squares).
        Sum of squared distances from each point to its assigned centroid.
        """
        # YOUR CODE HERE
        pass
