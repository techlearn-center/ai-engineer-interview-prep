# Learn: NumPy & Pandas for AI Engineers

NumPy and Pandas are the foundation of all data work in Python.
Every AI engineer uses these daily.

---

## Part A: NumPy

### 1. What is NumPy?

NumPy is a library for fast numerical operations on arrays.
Why not just use Python lists?

```python
# Python list: SLOW (loops through each element one by one)
result = []
for i in range(1000000):
    result.append(i * 2)

# NumPy: FAST (operates on all elements at once in C)
import numpy as np
result = np.arange(1000000) * 2  # 100x faster
```

**Key concept: Vectorization** - applying an operation to an entire array at once,
without writing a Python loop. This is how NumPy achieves its speed.

---

### 2. Creating Arrays

```python
import numpy as np

# From a list
a = np.array([1, 2, 3, 4, 5])

# Common constructors
np.zeros((3, 4))        # 3x4 matrix of zeros
np.ones((2, 3))         # 2x3 matrix of ones
np.arange(0, 10, 2)     # [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)    # [0, 0.25, 0.5, 0.75, 1.0]
np.random.randn(3, 2)   # 3x2 matrix of random normal values
np.eye(3)               # 3x3 identity matrix

# Check shape
a = np.array([[1, 2, 3], [4, 5, 6]])
a.shape     # (2, 3) - 2 rows, 3 columns
a.dtype     # dtype('int64')
a.ndim      # 2 (number of dimensions)
```

---

### 3. Indexing & Slicing

```python
a = np.array([[10, 20, 30],
              [40, 50, 60],
              [70, 80, 90]])

# Single element
a[0, 1]        # 20 (row 0, col 1)

# Slicing (row:col)
a[0, :]         # [10, 20, 30]  - entire first row
a[:, 1]         # [20, 50, 80]  - entire second column
a[0:2, 1:3]     # [[20, 30], [50, 60]]  - submatrix

# Boolean indexing (very common in ML!)
a[a > 50]       # [60, 70, 80, 90]  - all elements > 50

# Fancy indexing
indices = [0, 2]
a[indices]       # [[10, 20, 30], [70, 80, 90]]  - rows 0 and 2
```

---

### 4. Vectorized Operations (No Loops!)

This is the most important concept. NEVER loop over NumPy arrays.

```python
a = np.array([1, 2, 3, 4, 5])
b = np.array([10, 20, 30, 40, 50])

# Element-wise operations
a + b        # [11, 22, 33, 44, 55]
a * b        # [10, 40, 90, 160, 250]
a ** 2       # [1, 4, 9, 16, 25]
np.sqrt(a)   # [1.0, 1.414, 1.732, 2.0, 2.236]

# Aggregations
a.sum()      # 15
a.mean()     # 3.0
a.std()      # 1.414
a.min()      # 1
a.max()      # 5
a.argmax()   # 4 (INDEX of the max value)

# Along an axis (critical for 2D data!)
m = np.array([[1, 2], [3, 4], [5, 6]])
m.sum(axis=0)    # [9, 12]   - sum down columns
m.sum(axis=1)    # [3, 7, 11] - sum across rows
m.mean(axis=0)   # [3, 4]   - mean of each column
```

**Remember:** `axis=0` means "along rows" (collapse rows), `axis=1` means "along columns" (collapse columns).

---

### 5. Broadcasting

Broadcasting lets you operate on arrays of different shapes.

```python
# Scalar broadcast
a = np.array([1, 2, 3])
a * 10   # [10, 20, 30]  - multiplies each element

# Vector + Matrix
m = np.array([[1, 2, 3],
              [4, 5, 6]])
v = np.array([10, 20, 30])
m + v    # [[11, 22, 33], [14, 25, 36]]  - adds v to each row

# Column broadcast (reshape to column vector)
prices = np.array([100, 200, 300])      # shape (3,)
discounts = np.array([0.1, 0.2])         # shape (2,)

# To multiply: reshape so they broadcast
prices[:, None]   # shape (3, 1) - column vector
discounts[None, :] # shape (1, 2) - row vector

result = prices[:, None] * (1 - discounts[None, :])
# shape (3, 2) - every product x every discount
# [[90, 80], [180, 160], [270, 240]]
```

**Rule:** Broadcasting aligns shapes from the RIGHT. Dimensions must be equal or one of them must be 1.

---

### 6. Key ML Operations

```python
# Dot product (matrix multiplication)
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
np.dot(a, b)     # 32 (1*4 + 2*5 + 3*6)

# Matrix multiply
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
A @ B    # [[19, 22], [43, 50]]  (@ is matrix multiply)

# Norm (length of a vector)
np.linalg.norm(a)   # 3.742 (sqrt(1+4+9))

# Normalization (Z-score)
data = np.array([10, 20, 30, 40, 50])
normalized = (data - data.mean()) / data.std()
# mean=0, std=1

# Softmax (used in classification)
def softmax(x):
    e = np.exp(x - np.max(x))  # subtract max for stability
    return e / e.sum()
```

---

## Part B: Pandas

### 7. What is Pandas?

Pandas = "Excel in Python". It's for tabular data (rows and columns).

```python
import pandas as pd

# Create a DataFrame
df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "score": [90, 85, 95],
})
#       name  age  score
# 0    Alice   25     90
# 1      Bob   30     85
# 2  Charlie   35     95
```

---

### 8. Reading & Exploring Data

```python
# Read from file
df = pd.read_csv("data.csv")
df = pd.read_json("data.json")

# Explore
df.head()          # First 5 rows
df.tail()          # Last 5 rows
df.shape           # (n_rows, n_cols)
df.dtypes          # Data type of each column
df.describe()      # Statistics: mean, std, min, max, etc.
df.info()          # Column names, types, null counts
df.isnull().sum()  # Count of missing values per column
```

---

### 9. Selecting & Filtering

```python
# Select columns
df["name"]              # Single column (returns Series)
df[["name", "score"]]   # Multiple columns (returns DataFrame)

# Filter rows
df[df["age"] > 25]                    # Rows where age > 25
df[df["name"].isin(["Alice", "Bob"])] # Rows matching a list
df[(df["age"] > 25) & (df["score"] > 85)]  # Multiple conditions (use & not 'and')

# loc (label-based) and iloc (position-based)
df.loc[0, "name"]      # "Alice" (row label 0, column "name")
df.iloc[0, 0]          # "Alice" (row position 0, col position 0)
df.loc[df["age"] > 25, "name"]  # Names of people older than 25
```

---

### 10. GroupBy (Essential for Interviews)

GroupBy is the most tested Pandas operation.

```python
df = pd.DataFrame({
    "region": ["North", "South", "North", "South", "North"],
    "product": ["A", "A", "B", "B", "A"],
    "sales": [100, 150, 200, 80, 120],
})

# Basic groupby
df.groupby("region")["sales"].sum()
# North    420
# South    230

# Multiple aggregations
df.groupby("region")["sales"].agg(["sum", "mean", "count"])

# Group by multiple columns
df.groupby(["region", "product"])["sales"].sum()

# Reset index to get a regular DataFrame
result = df.groupby("region")["sales"].sum().reset_index()
```

---

### 11. Common Operations

```python
# Sorting
df.sort_values("score", ascending=False)

# Adding columns
df["pass"] = df["score"] >= 90

# Apply a function
df["name_upper"] = df["name"].apply(lambda x: x.upper())

# Handling missing values
df.dropna()                           # Remove rows with any NaN
df.fillna(0)                          # Fill NaN with 0
df["col"].fillna(df["col"].median())  # Fill with median

# Remove duplicates
df.drop_duplicates()
df.drop_duplicates(subset=["name"])  # Based on specific columns

# Pivot tables
pd.pivot_table(df, values="sales", index="region",
               columns="product", aggfunc="sum", fill_value=0)

# Rolling window (time series)
df["rolling_avg"] = df["sales"].rolling(window=7, min_periods=1).mean()

# Merge (like SQL JOIN)
df1.merge(df2, on="key_column", how="left")  # left, right, inner, outer
```

---

### 12. Pandas + NumPy Together

They work seamlessly:

```python
# DataFrame to NumPy array
X = df[["age", "score"]].values  # shape (n, 2) NumPy array

# NumPy array to DataFrame
arr = np.random.randn(100, 3)
df = pd.DataFrame(arr, columns=["feature1", "feature2", "feature3"])

# NumPy operations on DataFrame columns
df["normalized_score"] = (df["score"] - df["score"].mean()) / df["score"].std()
```

---

## Now Try the Problems

1. `p1_numpy_operations.py` - normalize, dot product, cosine similarity, softmax, broadcasting
2. `p2_pandas_wrangling.py` - groupby, rolling average, pivot tables, data cleaning

```bash
pytest 02_numpy_pandas/ -v
```
