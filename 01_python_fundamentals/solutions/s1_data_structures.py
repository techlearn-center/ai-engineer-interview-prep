"""
SOLUTIONS - Problem 1: Data Structures
========================================
Try to solve the problems yourself first!
Only look here if you're stuck for more than 15 minutes.
"""


def two_sum(nums: list[int], target: int) -> tuple[int, int]:
    """
    Key insight: Use a dictionary to store {value: index} as you iterate.
    For each number, check if (target - number) is already in the dict.
    Time: O(n), Space: O(n)
    """
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i


def group_anagrams(words: list[str]) -> list[list[str]]:
    """
    Key insight: Two words are anagrams if they have the same sorted characters.
    Use sorted characters as a dictionary key to group words.
    Time: O(n * k log k) where k is max word length
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for word in words:
        key = "".join(sorted(word))
        groups[key].append(word)

    # Sort each group and the outer list
    result = [sorted(group) for group in groups.values()]
    result.sort(key=lambda x: x[0])
    return result


def flatten_nested_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """
    Key insight: Recursion! If a value is a dict, recurse with updated parent_key.
    Otherwise, add to the result.
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_nested_dict(v, new_key, sep))
        else:
            items[new_key] = v
    return items


def lru_cache_dict(capacity: int) -> object:
    """
    Key insight: Python 3.7+ dicts maintain insertion order.
    Moving an item to the end = delete then re-insert.
    Evicting LRU = pop the first item.
    """
    class LRUCache:
        def __init__(self, cap):
            self.capacity = cap
            self.cache = {}

        def get(self, key):
            if key not in self.cache:
                return -1
            # Move to end (most recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            return value

        def put(self, key, value):
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) >= self.capacity:
                # Evict least recently used (first item)
                oldest_key = next(iter(self.cache))
                self.cache.pop(oldest_key)
            self.cache[key] = value

    return LRUCache(capacity)
