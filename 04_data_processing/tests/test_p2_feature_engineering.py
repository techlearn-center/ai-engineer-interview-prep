import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from p2_feature_engineering import (
    one_hot_encode,
    min_max_scale,
    add_polynomial_features,
    handle_missing_values,
)


class TestOneHotEncode:
    def test_basic(self):
        encoded, labels = one_hot_encode(["cat", "dog", "cat", "bird"])
        assert labels == ["bird", "cat", "dog"]
        assert encoded.shape == (4, 3)
        np.testing.assert_array_equal(encoded[0], [0, 1, 0])  # cat
        np.testing.assert_array_equal(encoded[3], [1, 0, 0])  # bird

    def test_single_class(self):
        encoded, labels = one_hot_encode(["a", "a", "a"])
        assert labels == ["a"]
        assert encoded.shape == (3, 1)


class TestMinMaxScale:
    def test_basic(self):
        X = np.array([[1, 10], [2, 20], [3, 30]])
        result = min_max_scale(X)
        np.testing.assert_array_almost_equal(result[0], [0, 0])
        np.testing.assert_array_almost_equal(result[2], [1, 1])

    def test_constant_column(self):
        X = np.array([[5, 1], [5, 2], [5, 3]])
        result = min_max_scale(X)
        assert np.all(result[:, 0] == 0)  # constant column -> 0

    def test_range(self):
        X = np.random.randn(50, 3)
        result = min_max_scale(X)
        assert result.min() >= 0
        assert result.max() <= 1


class TestPolynomialFeatures:
    def test_degree_2(self):
        X = np.array([[2], [3]])
        result = add_polynomial_features(X, degree=2)
        np.testing.assert_array_equal(result, [[2, 4], [3, 9]])

    def test_degree_3(self):
        X = np.array([[2], [3]])
        result = add_polynomial_features(X, degree=3)
        np.testing.assert_array_equal(result, [[2, 4, 8], [3, 9, 27]])

    def test_shape(self):
        X = np.array([[1], [2], [3], [4]])
        result = add_polynomial_features(X, degree=4)
        assert result.shape == (4, 4)


class TestHandleMissingValues:
    def test_mean(self):
        X = np.array([[1, 2], [np.nan, 4], [3, np.nan]])
        result = handle_missing_values(X, strategy="mean")
        assert not np.any(np.isnan(result))
        assert result[1, 0] == 2.0  # mean of 1, 3
        assert result[2, 1] == 3.0  # mean of 2, 4

    def test_median(self):
        X = np.array([[1, 2], [np.nan, 4], [5, np.nan]])
        result = handle_missing_values(X, strategy="median")
        assert not np.any(np.isnan(result))
        assert result[1, 0] == 3.0  # median of 1, 5

    def test_zero(self):
        X = np.array([[1, np.nan], [np.nan, 2]])
        result = handle_missing_values(X, strategy="zero")
        assert not np.any(np.isnan(result))
        assert result[0, 1] == 0
        assert result[1, 0] == 0
