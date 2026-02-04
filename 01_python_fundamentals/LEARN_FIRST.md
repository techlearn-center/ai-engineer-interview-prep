# Learn: Python Fundamentals for AI Engineers

These are the building blocks. If you're rusty, this will get you back up to speed.

---

## 1. Dictionaries (Hash Maps)

Dictionaries are the #1 most important data structure in Python interviews.
They give you O(1) lookup, insert, and delete.

```python
# Creating
d = {"name": "Alice", "age": 30}

# Accessing
d["name"]           # "Alice"
d.get("email", "")  # "" (default if key missing, avoids KeyError)

# Iterating
for key, value in d.items():
    print(key, value)

# Common patterns
d.setdefault("scores", []).append(90)  # Create list if key missing, then append

# Counter pattern (very common in interviews)
from collections import Counter
words = ["cat", "dog", "cat", "bird", "dog", "cat"]
counts = Counter(words)  # Counter({"cat": 3, "dog": 2, "bird": 1})

# defaultdict (auto-creates values for missing keys)
from collections import defaultdict
groups = defaultdict(list)
groups["animals"].append("cat")  # No KeyError!
```

**Key insight:** Whenever a problem says "find", "lookup", or "group" - think dictionary.

---

## 2. The Two-Sum Pattern

This is the most classic interview problem. It teaches the dictionary lookup pattern.

**Problem:** Given a list of numbers and a target, find two numbers that add up to target.

**Brute force (O(n^2)):** Check every pair.
```python
# DON'T do this in an interview
for i in range(len(nums)):
    for j in range(i+1, len(nums)):
        if nums[i] + nums[j] == target:
            return (i, j)
```

**Dictionary approach (O(n)):** For each number, check if its complement exists.
```python
def two_sum(nums, target):
    seen = {}  # value -> index
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:       # O(1) lookup!
            return (seen[complement], i)
        seen[num] = i                # Store for future lookups
```

**Why this matters:** This pattern (store what you've seen, check for what you need)
shows up in dozens of interview problems.

---

## 3. List Comprehensions

Python interviewers expect you to write Pythonic code. List comprehensions are essential.

```python
# Basic: transform every element
squares = [x**2 for x in range(10)]
# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# Filter: only include elements that match a condition
evens = [x for x in range(10) if x % 2 == 0]
# [0, 2, 4, 6, 8]

# Transform + filter combined
big_names = [name.upper() for name in names if len(name) > 3]

# Nested: flatten a 2D list
matrix = [[1, 2], [3, 4], [5, 6]]
flat = [x for row in matrix for x in row]
# [1, 2, 3, 4, 5, 6]

# Dict comprehension
word_lengths = {word: len(word) for word in ["cat", "elephant", "dog"]}
# {"cat": 3, "elephant": 8, "dog": 3}

# Set comprehension
unique_lengths = {len(word) for word in ["cat", "elephant", "dog"]}
# {3, 8}
```

**Interview tip:** If you write a for-loop that builds a list, the interviewer
will likely ask "Can you do that as a comprehension?"

---

## 4. Generators & yield

Generators produce values one at a time (lazily) instead of building a whole list in memory.

```python
# Regular function: builds entire list in memory
def get_squares_list(n):
    result = []
    for i in range(n):
        result.append(i**2)
    return result  # All in memory at once

# Generator: yields one value at a time
def get_squares_gen(n):
    for i in range(n):
        yield i**2  # Pauses here, resumes on next call

# Usage
for sq in get_squares_gen(1000000):
    print(sq)  # Only ONE value in memory at a time
```

**How yield works:**
1. Function runs until it hits `yield`
2. Returns the yielded value
3. Function PAUSES (remembers its state)
4. On next iteration, resumes from where it paused
5. Repeats until function ends

**When to use generators:**
- Processing large files line by line
- Sliding windows over data streams
- Any time you don't need all values at once
- Infinite sequences (e.g., Fibonacci)

```python
# Fibonacci generator - classic interview question
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

list(fibonacci(7))  # [0, 1, 1, 2, 3, 5, 8]
```

---

## 5. Decorators

A decorator is a function that wraps another function to add behavior.

**Why AI engineers need to know this:**
- FastAPI uses decorators: `@app.get("/predict")`
- PyTorch uses decorators: `@torch.no_grad()`
- Retry logic, caching, timing, logging - all use decorators

```python
# Step 1: Understand that functions are objects
def greet(name):
    return f"Hello {name}"

say_hi = greet          # Functions can be assigned to variables
say_hi("Alice")         # "Hello Alice"

# Step 2: Functions can take functions as arguments
def run_twice(func, arg):
    func(arg)
    func(arg)

# Step 3: A decorator wraps a function
import functools

def timer(func):
    @functools.wraps(func)          # Preserves original function name
    def wrapper(*args, **kwargs):   # Accepts any arguments
        import time
        start = time.time()
        result = func(*args, **kwargs)  # Call the original function
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper

# Using the decorator
@timer
def slow_function():
    import time
    time.sleep(1)
    return "done"

slow_function()  # Prints: "slow_function took 1.001s"
```

**Decorator with arguments (decorator factory):**
```python
def retry(max_attempts=3):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
        return wrapper
    return decorator

@retry(max_attempts=5)
def flaky_api_call():
    ...
```

**Mental model:** `@decorator` is just syntactic sugar for `func = decorator(func)`.

---

## 6. Useful Built-in Functions

Know these - they come up constantly:

```python
# enumerate: get index AND value
for i, name in enumerate(["Alice", "Bob", "Charlie"]):
    print(i, name)  # 0 Alice, 1 Bob, 2 Charlie

# zip: iterate two lists together
names = ["Alice", "Bob"]
scores = [90, 85]
for name, score in zip(names, scores):
    print(name, score)  # Alice 90, Bob 85

# zip for transpose
matrix = [[1, 2, 3], [4, 5, 6]]
transposed = list(zip(*matrix))  # [(1, 4), (2, 5), (3, 6)]

# sorted with key
students = [{"name": "Bob", "gpa": 3.5}, {"name": "Alice", "gpa": 3.9}]
sorted(students, key=lambda s: s["gpa"], reverse=True)

# any / all
nums = [1, 2, -3, 4]
any(n < 0 for n in nums)   # True (at least one negative)
all(n > 0 for n in nums)   # False (not all positive)

# map / filter (prefer comprehensions, but know these)
squared = list(map(lambda x: x**2, [1, 2, 3]))  # [1, 4, 9]
positive = list(filter(lambda x: x > 0, [-1, 2, -3, 4]))  # [2, 4]
```

---

## 7. String Methods to Know

```python
s = "Hello, World!"

s.lower()           # "hello, world!"
s.upper()           # "HELLO, WORLD!"
s.strip()           # Remove leading/trailing whitespace
s.split(",")        # ["Hello", " World!"]
",".join(["a","b"]) # "a,b"
s.replace("o", "0") # "Hell0, W0rld!"
s.startswith("He")  # True
s.endswith("!")     # True
"world" in s.lower() # True

# f-strings (know these cold)
name = "Alice"
score = 95.678
f"{name} scored {score:.1f}%"  # "Alice scored 95.7%"
```

---

## 8. Common Pitfalls

```python
# Mutable default arguments (CLASSIC gotcha)
def bad(items=[]):      # DON'T: same list shared across calls
    items.append(1)
    return items

def good(items=None):   # DO: use None and create new list
    if items is None:
        items = []
    items.append(1)
    return items

# Shallow vs deep copy
import copy
a = [[1, 2], [3, 4]]
b = a.copy()            # Shallow: inner lists are shared!
c = copy.deepcopy(a)    # Deep: completely independent

# Integer comparison
x = 256
y = 256
x is y   # True (Python caches small integers)
x == y   # True (always use == for value comparison)
```

---

## Now Try the Problems

1. `p1_data_structures.py` - two_sum, anagrams, flatten dict, LRU cache
2. `p2_comprehensions_generators.py` - matrix transpose, filtering, Fibonacci, sliding window
3. `p3_decorators_closures.py` - timer, retry, memoize, type validation

```bash
pytest 01_python_fundamentals/ -v
```
