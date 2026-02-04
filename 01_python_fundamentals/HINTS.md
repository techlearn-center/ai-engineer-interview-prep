# Hints - Python Fundamentals

## P1: Data Structures

### two_sum
- Think about what data structure lets you look up values in O(1) time
- As you iterate, store each number's index in a dictionary
- For each number, check if `target - number` exists in your dictionary

### group_anagrams
- What property do all anagrams share? (Think about sorted characters)
- Use `collections.defaultdict(list)` for grouping
- The key for grouping should be the sorted version of each word

### flatten_nested_dict
- This is a classic recursion problem
- Base case: value is NOT a dict -> add to result
- Recursive case: value IS a dict -> recurse with updated key prefix
- Build the new key with `f"{parent_key}{sep}{k}"`

### lru_cache_dict
- Python 3.7+ dicts maintain insertion order!
- "Most recently used" = move to END of dict (delete + re-insert)
- "Least recently used" = FIRST item in dict (`next(iter(dict))`)
- When capacity full: pop the first key before inserting new one

## P2: Comprehensions & Generators

### matrix_transpose
- `zip(*matrix)` is the magic trick - it unpacks rows and zips columns
- Remember to convert each tuple back to a list

### filter_and_transform
- Sort FIRST (by score descending), then filter, then transform
- `sorted(data, key=lambda x: x["score"], reverse=True)`
- List comprehension with `if` clause for filtering

### fibonacci_generator
- Use `yield` instead of `return`
- Keep two variables: current and next
- `a, b = b, a + b` is the Fibonacci update step

### sliding_window_avg
- Iterate from `i=0` to `len(data) - window_size`
- Slice `data[i:i+window_size]` for each window
- `yield sum(window) / len(window)`

## P3: Decorators & Closures

### timer_decorator
- `import functools` and use `@functools.wraps(func)`
- Record `time.time()` before and after calling `func`
- Store result as `wrapper.last_elapsed = elapsed`

### retry_decorator
- This is a decorator FACTORY: function -> decorator -> wrapper (3 levels of nesting)
- Use a for loop with `range(max_retries)`
- Catch the exception, save it, and if all retries fail, re-raise the last one

### memoize
- Use a dictionary `wrapper.cache = {}`
- Key = `args` tuple (it's hashable)
- Check if args already in cache before calling the function

### validate_types
- Use `inspect.signature(func)` to get parameter names
- Match positional args with parameter names by index
- Use `isinstance(arg, expected_type)` to check
