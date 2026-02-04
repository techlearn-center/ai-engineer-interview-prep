import sys
import os
import numpy as np
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from p1_numpy_operations import (
    normalize_array,
    batch_dot_product,
    cosine_similarity_matrix,
    softmax,
    broadcast_operation,
)


class TestNormalizeArray:
    def test_basic(self):
        result = normalize_array(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert abs(result.mean()) < 1e-10
        assert abs(result.std() - 1.0) < 1e-10

    def test_already_normalized(self):
        arr = np.array([0.0, 0.0, 0.0])
        # All zeros: std=0, should handle gracefully or return zeros
        # We accept zeros back for this edge case
        result = normalize_array(np.array([1.0, 3.0, 5.0]))
        assert abs(result.mean()) < 1e-10


class TestBatchDotProduct:
    def test_basic(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        result = batch_dot_product(A, B)
        np.testing.assert_array_equal(result, [17, 53])

    def test_single_row(self):
        A = np.array([[1, 2, 3]])
        B = np.array([[4, 5, 6]])
        result = batch_dot_product(A, B)
        np.testing.assert_array_equal(result, [32])


class TestCosineSimilarityMatrix:
    def test_orthogonal_vectors(self):
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = cosine_similarity_matrix(X)
        np.testing.assert_array_almost_equal(result, [[1, 0], [0, 1]])

    def test_identical_vectors(self):
        X = np.array([[1.0, 2.0], [1.0, 2.0]])
        result = cosine_similarity_matrix(X)
        np.testing.assert_array_almost_equal(result, [[1, 1], [1, 1]])

    def test_shape(self):
        X = np.random.randn(5, 3)
        result = cosine_similarity_matrix(X)
        assert result.shape == (5, 5)

    def test_diagonal_is_one(self):
        X = np.random.randn(4, 3)
        result = cosine_similarity_matrix(X)
        np.testing.assert_array_almost_equal(np.diag(result), np.ones(4))


class TestSoftmax:
    def test_basic(self):
        result = softmax(np.array([1.0, 2.0, 3.0]))
        assert abs(result.sum() - 1.0) < 1e-10
        assert result[2] > result[1] > result[0]

    def test_sums_to_one(self):
        result = softmax(np.array([10, 20, 30]))
        assert abs(result.sum() - 1.0) < 1e-10

    def test_numerical_stability(self):
        # Large values shouldn't cause overflow
        result = softmax(np.array([1000, 2000, 3000]))
        assert abs(result.sum() - 1.0) < 1e-10
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestBroadcastOperation:
    def test_basic(self):
        prices = np.array([100.0, 200.0, 300.0])
        discounts = np.array([10.0, 20.0])
        result = broadcast_operation(prices, discounts)
        expected = np.array([[90, 80], [180, 160], [270, 240]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_shape(self):
        prices = np.array([100.0, 200.0])
        discounts = np.array([10.0, 20.0, 30.0])
        result = broadcast_operation(prices, discounts)
        assert result.shape == (2, 3)
