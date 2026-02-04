import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from p1_linear_regression import LinearRegression


class TestLinearRegression:
    def setup_method(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 2)
        self.y = 3 * self.X[:, 0] + 5 * self.X[:, 1] + 2 + np.random.randn(100) * 0.1

    def test_fit_doesnt_crash(self):
        model = LinearRegression()
        model.fit(self.X, self.y, lr=0.01, epochs=1000)
        assert model.weights is not None
        assert model.bias is not None

    def test_weights_shape(self):
        model = LinearRegression()
        model.fit(self.X, self.y, lr=0.01, epochs=1000)
        assert model.weights.shape == (2,)

    def test_learns_correct_weights(self):
        model = LinearRegression()
        model.fit(self.X, self.y, lr=0.01, epochs=2000)
        # Should learn weights close to [3, 5] and bias close to 2
        np.testing.assert_almost_equal(model.weights[0], 3.0, decimal=0)
        np.testing.assert_almost_equal(model.weights[1], 5.0, decimal=0)
        np.testing.assert_almost_equal(model.bias, 2.0, decimal=0)

    def test_predict(self):
        model = LinearRegression()
        model.fit(self.X, self.y, lr=0.01, epochs=1000)
        y_pred = model.predict(self.X)
        assert y_pred.shape == self.y.shape

    def test_mse_is_low(self):
        model = LinearRegression()
        model.fit(self.X, self.y, lr=0.01, epochs=2000)
        y_pred = model.predict(self.X)
        mse = model.mse(self.y, y_pred)
        assert mse < 0.1  # should fit well since data is nearly linear

    def test_mse_known_value(self):
        assert LinearRegression.mse(
            np.array([1, 2, 3]),
            np.array([1, 2, 3])
        ) == 0.0

        assert abs(LinearRegression.mse(
            np.array([1, 2, 3]),
            np.array([2, 3, 4])
        ) - 1.0) < 1e-10
