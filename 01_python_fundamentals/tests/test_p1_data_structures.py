import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from p1_data_structures import two_sum, group_anagrams, flatten_nested_dict, lru_cache_dict


class TestTwoSum:
    def test_basic(self):
        assert two_sum([2, 7, 11, 15], 9) == (0, 1)

    def test_middle_elements(self):
        assert two_sum([3, 2, 4], 6) == (1, 2)

    def test_negative_numbers(self):
        assert two_sum([-1, -2, -3, -4, -5], -8) == (2, 4)

    def test_duplicates(self):
        assert two_sum([3, 3], 6) == (0, 1)


class TestGroupAnagrams:
    def test_basic(self):
        result = group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
        expected = [["ate", "eat", "tea"], ["bat"], ["nat", "tan"]]
        assert result == expected

    def test_empty_strings(self):
        result = group_anagrams(["", ""])
        assert result == [["", ""]]

    def test_single_words(self):
        result = group_anagrams(["abc", "def", "ghi"])
        assert result == [["abc"], ["def"], ["ghi"]]


class TestFlattenNestedDict:
    def test_basic(self):
        result = flatten_nested_dict({"a": 1, "b": {"c": 2, "d": {"e": 3}}})
        assert result == {"a": 1, "b.c": 2, "b.d.e": 3}

    def test_flat_dict(self):
        result = flatten_nested_dict({"x": 1, "y": 2})
        assert result == {"x": 1, "y": 2}

    def test_deeply_nested(self):
        result = flatten_nested_dict({"a": {"b": {"c": {"d": 42}}}})
        assert result == {"a.b.c.d": 42}

    def test_empty(self):
        assert flatten_nested_dict({}) == {}


class TestLRUCache:
    def test_basic_operations(self):
        cache = lru_cache_dict(2)
        cache.put(1, 1)
        cache.put(2, 2)
        assert cache.get(1) == 1
        cache.put(3, 3)  # evicts key 2
        assert cache.get(2) == -1

    def test_update_existing(self):
        cache = lru_cache_dict(2)
        cache.put(1, 1)
        cache.put(2, 2)
        cache.put(1, 10)  # update key 1
        assert cache.get(1) == 10
        cache.put(3, 3)  # should evict key 2, not 1
        assert cache.get(2) == -1
        assert cache.get(1) == 10

    def test_capacity_one(self):
        cache = lru_cache_dict(1)
        cache.put(1, 1)
        cache.put(2, 2)  # evicts key 1
        assert cache.get(1) == -1
        assert cache.get(2) == 2
