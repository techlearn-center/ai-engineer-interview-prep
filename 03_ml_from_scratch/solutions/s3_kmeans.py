"""
SOLUTIONS - K-Means Clustering from Scratch
==============================================
Try to solve the problems yourself first!
"""
import numpy as np


class KMeans:
    """
    Key concepts to explain in interview:
    - K-Means is unsupervised - no labels needed
    - It alternates between two steps:
        1. ASSIGN: each point goes to nearest centroid
        2. UPDATE: move centroids to mean of their assigned points
    - Converges when assignments don't change
    - Sensitive to initialization (that's why K-Means++ exists)
    """

    def __init__(self, k: int = 3, max_iters: int = 100, random_state: int = 42):
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    def fit(self, X: np.ndarray):
        np.random.seed(self.random_state)
        n_samples = X.shape[0]

        # Step 1: Random initialization - pick k random data points
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices].copy()

        for _ in range(self.max_iters):
            # Step 2a: Assign each point to nearest centroid
            old_labels = self.labels
            self.labels = self._assign_clusters(X)

            # Step 2b: Update centroids
            self.centroids = self._update_centroids(X, self.labels)

            # Check convergence
            if old_labels is not None and np.all(old_labels == self.labels):
                break

    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        """
        Key insight: Use broadcasting to compute ALL distances at once.
        X[:, None, :] has shape (n, 1, d)
        centroids[None, :, :] has shape (1, k, d)
        Difference: (n, k, d) -> square -> sum over d -> (n, k)
        Then argmin over k gives the nearest centroid for each point.
        """
        # Compute distance from each point to each centroid
        distances = np.sqrt(((X[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute mean of each cluster's points."""
        new_centroids = np.zeros_like(self.centroids)
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(axis=0)
            else:
                # Empty cluster: keep old centroid
                new_centroids[i] = self.centroids[i]
        return new_centroids

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._assign_clusters(X)

    @staticmethod
    def inertia(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        """Sum of squared distances from each point to its centroid."""
        total = 0.0
        for i in range(len(centroids)):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                total += np.sum((cluster_points - centroids[i]) ** 2)
        return total
