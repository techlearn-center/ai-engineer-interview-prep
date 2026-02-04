"""
Problem 3: Decorators & Closures
=================================
Difficulty: Medium

Decorators come up often in AI interviews because frameworks like
FastAPI, Flask, PyTorch, and TensorFlow use them heavily.

Run tests:
    pytest 01_python_fundamentals/tests/test_p3_decorators_closures.py -v
"""
import time
import functools


def timer_decorator(func):
    """
    Write a decorator that measures execution time of a function.
    Store the elapsed time as an attribute on the wrapper: wrapper.last_elapsed

    Example:
        @timer_decorator
        def slow_func():
            time.sleep(0.1)

        slow_func()
        print(slow_func.last_elapsed)  # ~0.1
    """
    # YOUR CODE HERE
    pass


def retry_decorator(max_retries: int = 3):
    """
    Write a decorator FACTORY (takes arguments) that retries a function
    up to max_retries times if it raises an exception.
    If all retries fail, raise the last exception.

    Example:
        @retry_decorator(max_retries=2)
        def flaky_func():
            ...
    """
    # YOUR CODE HERE
    pass


def memoize(func):
    """
    Write a memoization decorator that caches function results.
    The decorator should work with any hashable arguments.
    Store the cache as an attribute: wrapper.cache

    Example:
        @memoize
        def expensive(n):
            return n ** 2

        expensive(5)  # computes
        expensive(5)  # returns cached
        expensive.cache  # {(5,): 25}
    """
    # YOUR CODE HERE
    pass


def validate_types(**type_hints):
    """
    Write a decorator factory that validates argument types at runtime.
    Raise TypeError with a descriptive message if types don't match.

    Example:
        @validate_types(x=int, y=str)
        def greet(x, y):
            return f"{y} * {x}"

        greet(3, "hi")  # works
        greet("3", "hi")  # raises TypeError
    """
    # YOUR CODE HERE
    pass
