# Python Syntax Cheatsheet

Print this or keep it open during practice. Covers everything you'll type in an interview.

---

## Data Structures

```python
# List
items = [1, 2, 3]
items.append(4)             # [1, 2, 3, 4]
items.insert(0, 0)          # [0, 1, 2, 3, 4]
items.pop()                 # removes & returns last: 4
items.pop(0)                # removes & returns first: 0
items.remove(2)             # removes first occurrence of 2
items.sort()                # in-place sort
sorted(items)               # returns new sorted list
items.reverse()             # in-place reverse
len(items)                  # length
items[-1]                   # last element
items[1:3]                  # slice [1, 2]
items[::-1]                 # reversed copy

# Dictionary
d = {"a": 1, "b": 2}
d["c"] = 3                  # add/update
d.get("x", 0)               # get with default (no KeyError)
d.pop("a")                   # remove and return value
d.keys()                     # dict_keys(["b", "c"])
d.values()                   # dict_values([2, 3])
d.items()                    # dict_items([("b", 2), ("c", 3)])
"b" in d                     # True (key lookup is O(1))
d.setdefault("x", [])       # get or create with default

# Set
s = {1, 2, 3}
s.add(4)
s.remove(1)                  # KeyError if missing
s.discard(1)                 # no error if missing
s1 & s2                      # intersection
s1 | s2                      # union
s1 - s2                      # difference

# Tuple (immutable)
t = (1, 2, 3)
a, b, c = t                 # unpacking

# defaultdict & Counter
from collections import defaultdict, Counter
dd = defaultdict(list)       # dd["new_key"] auto-creates []
dd = defaultdict(int)        # dd["new_key"] auto-creates 0
c = Counter(["a", "b", "a"]) # Counter({"a": 2, "b": 1})
c.most_common(2)             # [("a", 2), ("b", 1)]
```

---

## String Operations

```python
s = "Hello World"
s.lower()                    # "hello world"
s.upper()                    # "HELLO WORLD"
s.strip()                    # remove whitespace from both ends
s.split()                    # ["Hello", "World"]
s.split(",")                 # split on comma
",".join(["a", "b", "c"])   # "a,b,c"
s.replace("o", "0")         # "Hell0 W0rld"
s.startswith("He")           # True
s.endswith("ld")             # True
s.find("World")              # 6 (index, -1 if not found)
s.count("l")                 # 3
f"{name} is {age}"           # f-string formatting
f"{3.14159:.2f}"             # "3.14"
f"{42:05d}"                  # "00042"
```

---

## Comprehensions

```python
# List comprehension
[x**2 for x in range(5)]                        # [0, 1, 4, 9, 16]
[x for x in range(10) if x % 2 == 0]            # [0, 2, 4, 6, 8]
[x.upper() for x in names if len(x) > 3]

# Dict comprehension
{k: v**2 for k, v in d.items()}
{word: len(word) for word in words}

# Set comprehension
{len(word) for word in words}

# Nested comprehension
[x for row in matrix for x in row]               # flatten
[[row[i] for row in matrix] for i in range(cols)] # transpose
```

---

## Functions & Lambda

```python
# Regular function
def add(a: int, b: int = 0) -> int:
    return a + b

# Lambda (anonymous function)
square = lambda x: x**2
sorted(items, key=lambda x: x["score"])

# *args and **kwargs
def func(*args, **kwargs):
    print(args)    # tuple of positional args
    print(kwargs)  # dict of keyword args

# Type hints
def process(data: list[dict], threshold: float = 0.5) -> list[str]:
    ...
```

---

## Loops & Iteration

```python
# enumerate (get index + value)
for i, item in enumerate(items):
    print(i, item)

# zip (iterate multiple lists)
for name, score in zip(names, scores):
    print(name, score)

# dict iteration
for key, value in d.items():
    print(key, value)

# range
range(5)          # 0, 1, 2, 3, 4
range(2, 8)       # 2, 3, 4, 5, 6, 7
range(0, 10, 2)   # 0, 2, 4, 6, 8

# any / all
any(x > 0 for x in nums)    # True if any positive
all(x > 0 for x in nums)    # True if ALL positive
```

---

## Error Handling

```python
try:
    result = risky_operation()
except KeyError as e:
    print(f"Key not found: {e}")
except (ValueError, TypeError) as e:
    print(f"Bad value: {e}")
except Exception as e:
    print(f"Unexpected: {e}")
finally:
    cleanup()

# Raise your own
raise ValueError(f"Expected positive, got {x}")
raise KeyError(f"Model '{name}' not found")
```

---

## Classes

```python
class Model:
    def __init__(self, name: str, version: str = "v1"):
        self.name = name
        self.version = version
        self._weights = None          # convention: "private"

    def train(self, X, y):
        self._weights = ...
        return self

    def predict(self, X):
        if self._weights is None:
            raise RuntimeError("Model not trained")
        return X @ self._weights

    def __repr__(self):
        return f"Model(name={self.name}, version={self.version})"

# Dataclass (quick way to define data-holding classes)
from dataclasses import dataclass

@dataclass
class Config:
    lr: float = 0.01
    epochs: int = 100
    batch_size: int = 32
```

---

## File I/O

```python
# Read
with open("file.txt") as f:
    content = f.read()          # entire file as string
    lines = f.readlines()       # list of lines

# Write
with open("file.txt", "w") as f:
    f.write("hello\n")

# JSON
import json
with open("data.json") as f:
    data = json.load(f)         # file -> dict

json.dumps(data)                # dict -> string
json.loads(string)              # string -> dict

# CSV
import csv
with open("data.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row)              # each row is a dict
```

---

## Decorators

```python
import functools

# Basic decorator
def my_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # before
        result = func(*args, **kwargs)
        # after
        return result
    return wrapper

# Decorator with arguments
def repeat(n):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(n):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def say_hi():
    print("hi")
```

---

## Generators

```python
def my_generator(n):
    for i in range(n):
        yield i * 2          # pauses here, resumes on next()

# Usage
for val in my_generator(5):
    print(val)                # 0, 2, 4, 6, 8

list(my_generator(5))         # [0, 2, 4, 6, 8]

# Generator expression
squares = (x**2 for x in range(1000000))  # lazy, no memory
```

---

## Regex

```python
import re

re.sub(r'pattern', 'replacement', text)    # find & replace
re.findall(r'pattern', text)                # all matches
re.search(r'pattern', text)                 # first match
re.match(r'pattern', text)                  # match at start

# Common patterns
r'\d+'          # one or more digits
r'\w+'          # one or more word chars
r'\s+'          # one or more whitespace
r'[^a-z]'       # not lowercase letter
r'https?://\S+' # URLs
r'\S+@\S+\.\S+' # emails
```

---

## NumPy Quick Reference

```python
import numpy as np

# Create
np.array([1, 2, 3])
np.zeros((3, 4))
np.ones((2, 3))
np.random.randn(3, 2)
np.arange(0, 10, 2)
np.eye(3)

# Shape
a.shape, a.dtype, a.ndim

# Operations
a + b, a * b, a @ b          # element-wise, element-wise, matrix multiply
a.sum(), a.mean(), a.std()
a.sum(axis=0)                 # sum each column
a.sum(axis=1)                 # sum each row
a.argmax(), a.argmin()        # index of max/min
np.dot(a, b)                  # dot product
np.linalg.norm(a)             # vector length

# Reshape & index
a.reshape(3, 2)
a[:, None]                    # add axis (for broadcasting)
a[a > 0]                      # boolean indexing
np.clip(a, -1, 1)             # clamp values

# Key formulas
(a - a.mean()) / a.std()      # Z-score normalize
np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()  # softmax
np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))  # cosine sim
```

---

## Pandas Quick Reference

```python
import pandas as pd

# Create & read
df = pd.DataFrame({"col1": [1,2], "col2": [3,4]})
df = pd.read_csv("file.csv")

# Explore
df.head(), df.shape, df.dtypes, df.describe()
df.isnull().sum()

# Select
df["col"]                     # one column
df[["col1", "col2"]]          # multiple columns
df[df["col"] > 5]             # filter rows
df.loc[0, "col"]              # by label
df.iloc[0, 0]                 # by position

# Group & aggregate
df.groupby("col")["value"].sum()
df.groupby("col")["value"].agg(["sum", "mean", "count"])
df.groupby(["a", "b"])["c"].sum().reset_index()

# Transform
df.sort_values("col", ascending=False)
df["new"] = df["old"].apply(lambda x: x * 2)
df.drop_duplicates()
df.fillna(0)
df["col"].rolling(window=7, min_periods=1).mean()

# Pivot
pd.pivot_table(df, values="v", index="row", columns="col", aggfunc="sum", fill_value=0)

# Merge
df1.merge(df2, on="key", how="left")
```
