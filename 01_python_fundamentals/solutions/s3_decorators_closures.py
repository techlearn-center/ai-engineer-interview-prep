"""
SOLUTIONS - Problem 3: Decorators & Closures
==============================================
Try to solve the problems yourself first!
"""
import time
import functools


def timer_decorator(func):
    """
    Key insight: @functools.wraps preserves the original function's name/docs.
    Store the elapsed time as an attribute on the wrapper function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        wrapper.last_elapsed = time.time() - start
        return result
    wrapper.last_elapsed = 0
    return wrapper


def retry_decorator(max_retries: int = 3):
    """
    Key insight: A decorator factory returns a decorator.
    It's a function that returns a function that returns a function.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
            raise last_exception
        return wrapper
    return decorator


def memoize(func):
    """
    Key insight: Use a dictionary to cache results keyed by arguments.
    args tuple works as a dict key since it's hashable.
    """
    @functools.wraps(func)
    def wrapper(*args):
        if args not in wrapper.cache:
            wrapper.cache[args] = func(*args)
        return wrapper.cache[args]
    wrapper.cache = {}
    return wrapper


def validate_types(**type_hints):
    """
    Key insight: Use inspect or zip to match parameter names with arguments.
    Check each argument against the expected type.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function parameter names
            import inspect
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())

            # Check positional args
            for i, arg in enumerate(args):
                param_name = param_names[i]
                if param_name in type_hints:
                    expected_type = type_hints[param_name]
                    if not isinstance(arg, expected_type):
                        raise TypeError(
                            f"Argument '{param_name}' expected {expected_type.__name__}, "
                            f"got {type(arg).__name__}"
                        )

            # Check keyword args
            for name, arg in kwargs.items():
                if name in type_hints:
                    expected_type = type_hints[name]
                    if not isinstance(arg, expected_type):
                        raise TypeError(
                            f"Argument '{name}' expected {expected_type.__name__}, "
                            f"got {type(arg).__name__}"
                        )

            return func(*args, **kwargs)
        return wrapper
    return decorator
