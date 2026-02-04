"""
Problem 1: Python Data Structures
==================================
Difficulty: Easy -> Medium

These problems test your ability to work with Python's core data structures.
Fill in each function and run the tests to check your answers:

    pytest 01_python_fundamentals/tests/test_p1_data_structures.py -v
"""


def two_sum(nums: list[int], target: int) -> tuple[int, int]:
    """
    Given a list of integers and a target sum, return the INDICES of the
    two numbers that add up to the target.

    You may assume exactly one solution exists.
    Use a dictionary/hashmap for O(n) time complexity.

    Example:
        two_sum([2, 7, 11, 15], 9) -> (0, 1)
        two_sum([3, 2, 4], 6) -> (1, 2)
    """
    # YOUR CODE HERE
    pass


def group_anagrams(words: list[str]) -> list[list[str]]:
    """
    Given a list of strings, group the anagrams together.
    Return a list of groups (each group is a sorted list).
    The outer list should be sorted by the first element of each group.

    Example:
        group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
        -> [["ate", "eat", "tea"], ["bat"], ["nat", "tan"]]
    """
    # YOUR CODE HERE
    pass


def flatten_nested_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """
    Flatten a nested dictionary using dot notation for keys.

    Example:
        flatten_nested_dict({"a": 1, "b": {"c": 2, "d": {"e": 3}}})
        -> {"a": 1, "b.c": 2, "b.d.e": 3}
    """
    # YOUR CODE HERE
    pass


def lru_cache_dict(capacity: int) -> object:
    """
    Implement a simple LRU (Least Recently Used) cache using a dictionary.
    Return an object/class instance that supports:
        - get(key) -> returns value or -1 if not found
        - put(key, value) -> inserts or updates the key-value pair
    When the cache exceeds capacity, remove the least recently used item.

    Hint: Python 3.7+ dicts maintain insertion order. Use this property.

    Example:
        cache = lru_cache_dict(2)
        cache.put(1, 1)
        cache.put(2, 2)
        cache.get(1)       -> 1
        cache.put(3, 3)    # evicts key 2
        cache.get(2)       -> -1
    """
    # YOUR CODE HERE
    pass
