"""
SOLUTIONS - Problem 2: Comprehensions & Generators
====================================================
Try to solve the problems yourself first!
"""


def matrix_transpose(matrix: list[list[int]]) -> list[list[int]]:
    """
    Key insight: zip(*matrix) unpacks the rows and zips columns together.
    """
    return [list(row) for row in zip(*matrix)]


def filter_and_transform(data: list[dict]) -> list[str]:
    """
    Key insight: Sort first, then filter, then transform in a comprehension.
    """
    sorted_data = sorted(data, key=lambda x: x["score"], reverse=True)
    return [d["name"].upper() for d in sorted_data if d["score"] >= 80]


def fibonacci_generator(n: int):
    """
    Key insight: yield pauses the function and returns a value.
    The function resumes from where it left off on the next call.
    """
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b


def sliding_window_avg(data: list[float], window_size: int):
    """
    Key insight: Iterate through valid start positions.
    For each position, compute the average of the window.
    """
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        yield sum(window) / len(window)


def nested_dict_comprehension(keys: list[str], values: list[list[int]]) -> dict:
    """
    Key insight: zip keys with values, compute sum, filter with condition.
    """
    return {k: sum(v) for k, v in zip(keys, values) if sum(v) > 0}
