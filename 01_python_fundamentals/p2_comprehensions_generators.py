"""
Problem 2: List Comprehensions & Generators
============================================
Difficulty: Easy -> Medium

These are VERY common in AI engineer interviews. You'll be expected to
write Pythonic one-liners and understand lazy evaluation.

Run tests:
    pytest 01_python_fundamentals/tests/test_p2_comprehensions_generators.py -v
"""


def matrix_transpose(matrix: list[list[int]]) -> list[list[int]]:
    """
    Transpose a matrix using a list comprehension (one-liner).

    Example:
        matrix_transpose([[1, 2, 3], [4, 5, 6]])
        -> [[1, 4], [2, 5], [3, 6]]
    """
    # YOUR CODE HERE (try to do it in one line)
    pass


def filter_and_transform(data: list[dict]) -> list[str]:
    """
    Given a list of dicts with 'name' and 'score' keys, return a list of
    uppercase names for entries where score >= 80, sorted by score descending.

    Use a list comprehension.

    Example:
        data = [
            {"name": "alice", "score": 90},
            {"name": "bob", "score": 70},
            {"name": "charlie", "score": 85},
        ]
        -> ["ALICE", "CHARLIE"]
    """
    # YOUR CODE HERE
    pass


def fibonacci_generator(n: int):
    """
    Create a generator that yields the first n Fibonacci numbers.
    Use the yield keyword.

    Example:
        list(fibonacci_generator(7)) -> [0, 1, 1, 2, 3, 5, 8]
    """
    # YOUR CODE HERE
    pass


def sliding_window_avg(data: list[float], window_size: int):
    """
    Create a generator that yields the average of a sliding window
    over the data. This is commonly used in time-series / ML preprocessing.

    Example:
        list(sliding_window_avg([1, 2, 3, 4, 5], 3))
        -> [2.0, 3.0, 4.0]
    """
    # YOUR CODE HERE
    pass


def nested_dict_comprehension(keys: list[str], values: list[list[int]]) -> dict:
    """
    Create a dict comprehension where each key maps to the sum of its values,
    but only include entries where the sum is positive.

    Example:
        nested_dict_comprehension(
            ["a", "b", "c"],
            [[1, 2, 3], [-5, -10, 2], [0, 0, 1]]
        ) -> {"a": 6, "c": 1}
    """
    # YOUR CODE HERE
    pass
