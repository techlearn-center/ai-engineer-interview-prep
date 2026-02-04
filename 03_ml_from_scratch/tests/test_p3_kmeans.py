import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from p3_kmeans import KMeans


class TestKMeans:
    def setup_method(self):
        np.random.seed(42)
        # 3 clear clusters
        c1 = np.random.randn(30, 2) + np.array([0, 0])
        c2 = np.random.randn(30, 2) + np.array([10, 10])
        c3 = np.random.randn(30, 2) + np.array([20, 0])
        self.X = np.vstack([c1, c2, c3])

    def test_fit_doesnt_crash(self):
        model = KMeans(k=3)
        model.fit(self.X)
        assert model.centroids is not None
        assert model.labels is not None

    def test_centroids_shape(self):
        model = KMeans(k=3)
        model.fit(self.X)
        assert model.centroids.shape == (3, 2)

    def test_labels_shape(self):
        model = KMeans(k=3)
        model.fit(self.X)
        assert model.labels.shape == (90,)

    def test_finds_three_clusters(self):
        model = KMeans(k=3)
        model.fit(self.X)
        assert len(np.unique(model.labels)) == 3

    def test_correct_clustering(self):
        model = KMeans(k=3)
        model.fit(self.X)
        # Points from each group should have the same label
        labels_c1 = model.labels[:30]
        labels_c2 = model.labels[30:60]
        labels_c3 = model.labels[60:]
        # Each cluster should be mostly one label
        assert len(np.unique(labels_c1)) == 1
        assert len(np.unique(labels_c2)) == 1
        assert len(np.unique(labels_c3)) == 1
        # All three labels should be different
        assert len({labels_c1[0], labels_c2[0], labels_c3[0]}) == 3

    def test_predict(self):
        model = KMeans(k=3)
        model.fit(self.X)
        new_points = np.array([[0, 0], [10, 10], [20, 0]])
        preds = model.predict(new_points)
        assert preds.shape == (3,)
        assert len(np.unique(preds)) == 3  # each close to a different cluster

    def test_inertia(self):
        model = KMeans(k=3)
        model.fit(self.X)
        inertia = KMeans.inertia(self.X, model.labels, model.centroids)
        assert inertia > 0
        assert inertia < 500  # well-separated clusters should have low inertia
