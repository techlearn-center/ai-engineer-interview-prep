import sys
import os
import time
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from p3_decorators_closures import timer_decorator, retry_decorator, memoize, validate_types


class TestTimerDecorator:
    def test_measures_time(self):
        @timer_decorator
        def slow():
            time.sleep(0.05)
            return "done"

        result = slow()
        assert result == "done"
        assert slow.last_elapsed >= 0.04

    def test_preserves_function_name(self):
        @timer_decorator
        def my_func():
            pass
        assert my_func.__name__ == "my_func"


class TestRetryDecorator:
    def test_succeeds_first_try(self):
        call_count = 0

        @retry_decorator(max_retries=3)
        def good_func():
            nonlocal call_count
            call_count += 1
            return "ok"

        assert good_func() == "ok"
        assert call_count == 1

    def test_retries_then_succeeds(self):
        call_count = 0

        @retry_decorator(max_retries=3)
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("fail")
            return "ok"

        assert flaky() == "ok"
        assert call_count == 3

    def test_all_retries_fail(self):
        @retry_decorator(max_retries=2)
        def always_fail():
            raise RuntimeError("nope")

        with pytest.raises(RuntimeError, match="nope"):
            always_fail()


class TestMemoize:
    def test_caches_result(self):
        call_count = 0

        @memoize
        def square(n):
            nonlocal call_count
            call_count += 1
            return n ** 2

        assert square(5) == 25
        assert square(5) == 25
        assert call_count == 1

    def test_cache_attribute(self):
        @memoize
        def add(a, b):
            return a + b

        add(1, 2)
        assert (1, 2) in add.cache
        assert add.cache[(1, 2)] == 3


class TestValidateTypes:
    def test_valid_types(self):
        @validate_types(x=int, y=str)
        def greet(x, y):
            return f"{y} * {x}"

        assert greet(3, "hi") == "hi * 3"

    def test_invalid_type_raises(self):
        @validate_types(x=int, y=str)
        def greet(x, y):
            return f"{y} * {x}"

        with pytest.raises(TypeError):
            greet("3", "hi")

    def test_partial_validation(self):
        @validate_types(x=int)
        def func(x, y):
            return x + y

        assert func(1, 2) == 3  # y is not validated
