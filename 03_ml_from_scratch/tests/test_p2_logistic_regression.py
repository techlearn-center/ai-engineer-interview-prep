import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from p2_logistic_regression import LogisticRegression


class TestSigmoid:
    def test_zero(self):
        assert LogisticRegression.sigmoid(np.array([0.0]))[0] == 0.5

    def test_large_positive(self):
        result = LogisticRegression.sigmoid(np.array([100.0]))
        assert abs(result[0] - 1.0) < 1e-5

    def test_large_negative(self):
        result = LogisticRegression.sigmoid(np.array([-100.0]))
        assert abs(result[0]) < 1e-5

    def test_no_nan(self):
        result = LogisticRegression.sigmoid(np.array([1000, -1000]))
        assert not np.any(np.isnan(result))


class TestLogisticRegression:
    def setup_method(self):
        np.random.seed(42)
        # Linearly separable data
        X_pos = np.random.randn(50, 2) + np.array([2, 2])
        X_neg = np.random.randn(50, 2) + np.array([-2, -2])
        self.X = np.vstack([X_pos, X_neg])
        self.y = np.array([1] * 50 + [0] * 50, dtype=float)

    def test_fit_doesnt_crash(self):
        model = LogisticRegression()
        model.fit(self.X, self.y, lr=0.1, epochs=1000)
        assert model.weights is not None

    def test_predict_proba_range(self):
        model = LogisticRegression()
        model.fit(self.X, self.y, lr=0.1, epochs=1000)
        probs = model.predict_proba(self.X)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_predict_labels(self):
        model = LogisticRegression()
        model.fit(self.X, self.y, lr=0.1, epochs=1000)
        preds = model.predict(self.X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_high_accuracy(self):
        model = LogisticRegression()
        model.fit(self.X, self.y, lr=0.1, epochs=2000)
        preds = model.predict(self.X)
        acc = model.accuracy(self.y, preds)
        assert acc > 0.9  # should easily separate this data

    def test_accuracy_known(self):
        assert LogisticRegression.accuracy(
            np.array([1, 1, 0, 0]),
            np.array([1, 0, 0, 0])
        ) == 0.75
