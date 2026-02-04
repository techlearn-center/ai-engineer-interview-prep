import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from p2_comprehensions_generators import (
    matrix_transpose,
    filter_and_transform,
    fibonacci_generator,
    sliding_window_avg,
    nested_dict_comprehension,
)


class TestMatrixTranspose:
    def test_basic(self):
        assert matrix_transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]

    def test_square(self):
        assert matrix_transpose([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]

    def test_single_row(self):
        assert matrix_transpose([[1, 2, 3]]) == [[1], [2], [3]]


class TestFilterAndTransform:
    def test_basic(self):
        data = [
            {"name": "alice", "score": 90},
            {"name": "bob", "score": 70},
            {"name": "charlie", "score": 85},
        ]
        assert filter_and_transform(data) == ["ALICE", "CHARLIE"]

    def test_all_pass(self):
        data = [
            {"name": "x", "score": 100},
            {"name": "y", "score": 80},
        ]
        assert filter_and_transform(data) == ["X", "Y"]

    def test_none_pass(self):
        data = [{"name": "x", "score": 50}]
        assert filter_and_transform(data) == []


class TestFibonacciGenerator:
    def test_seven(self):
        assert list(fibonacci_generator(7)) == [0, 1, 1, 2, 3, 5, 8]

    def test_one(self):
        assert list(fibonacci_generator(1)) == [0]

    def test_two(self):
        assert list(fibonacci_generator(2)) == [0, 1]

    def test_is_generator(self):
        import types
        assert isinstance(fibonacci_generator(5), types.GeneratorType)


class TestSlidingWindowAvg:
    def test_basic(self):
        result = list(sliding_window_avg([1, 2, 3, 4, 5], 3))
        assert result == [2.0, 3.0, 4.0]

    def test_window_one(self):
        result = list(sliding_window_avg([10, 20, 30], 1))
        assert result == [10.0, 20.0, 30.0]

    def test_window_equals_length(self):
        result = list(sliding_window_avg([1, 2, 3], 3))
        assert result == [2.0]

    def test_is_generator(self):
        import types
        assert isinstance(sliding_window_avg([1, 2], 1), types.GeneratorType)


class TestNestedDictComprehension:
    def test_basic(self):
        result = nested_dict_comprehension(
            ["a", "b", "c"],
            [[1, 2, 3], [-5, -10, 2], [0, 0, 1]]
        )
        assert result == {"a": 6, "c": 1}

    def test_all_positive(self):
        result = nested_dict_comprehension(["x", "y"], [[1, 1], [2, 2]])
        assert result == {"x": 2, "y": 4}

    def test_all_negative(self):
        result = nested_dict_comprehension(["x"], [[-1, -2]])
        assert result == {}
