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

## Part C: Pandas Deep Dive (Interview Focus)

The sections above cover basics. This section goes deeper into patterns you'll see in interviews.

### 13. Reading Files (All Formats)

```python
import pandas as pd

# CSV - most common
df = pd.read_csv("data.csv")
df = pd.read_csv("data.csv", sep=";")              # Custom separator
df = pd.read_csv("data.csv", usecols=["name", "age"])  # Only certain columns
df = pd.read_csv("data.csv", nrows=100)            # First 100 rows only
df = pd.read_csv("data.csv", skiprows=5)           # Skip first 5 rows
df = pd.read_csv("data.csv", na_values=["N/A", ""])  # Custom NA values

# JSON
df = pd.read_json("data.json")
df = pd.read_json("data.json", orient="records")   # List of dicts

# Excel
df = pd.read_excel("data.xlsx", sheet_name="Sheet1")

# From a URL (works with CSV, JSON, etc.)
df = pd.read_csv("https://example.com/data.csv")

# Writing files
df.to_csv("output.csv", index=False)               # Don't include row numbers
df.to_json("output.json", orient="records")
df.to_excel("output.xlsx", index=False)
```

---

### 14. Data Types & Conversion

```python
# Check types
df.dtypes
df["column"].dtype

# Convert types
df["age"] = df["age"].astype(int)
df["price"] = df["price"].astype(float)
df["is_active"] = df["is_active"].astype(bool)

# Parse dates (VERY common in interviews)
df["date"] = pd.to_datetime(df["date"])
df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
df["date"] = pd.to_datetime(df["date"], errors="coerce")  # Invalid -> NaT

# Extract from dates
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["weekday"] = df["date"].dt.dayofweek  # 0=Monday
df["quarter"] = df["date"].dt.quarter

# Categories (more memory efficient for low-cardinality columns)
df["status"] = df["status"].astype("category")
```

---

### 15. String Operations

```python
# String accessor: .str
df["name"].str.lower()              # Lowercase
df["name"].str.upper()              # Uppercase
df["name"].str.strip()              # Remove whitespace
df["name"].str.replace("old", "new")
df["name"].str.contains("pattern")  # Returns boolean Series
df["name"].str.startswith("A")
df["name"].str.split(" ")           # Split into lists
df["name"].str.len()                # String length

# Extract patterns with regex
df["email"].str.extract(r"@(.+)")   # Extract domain

# Combine columns
df["full_name"] = df["first"] + " " + df["last"]

# Example: Clean and standardize names
df["name"] = df["name"].str.strip().str.title()
```

---

### 16. Handling Missing Data (Critical!)

```python
# Detecting missing values
df.isnull()                     # Boolean DataFrame
df.isnull().sum()               # Count per column
df.isnull().sum().sum()         # Total missing
df.notnull()                    # Opposite of isnull

# Dropping missing values
df.dropna()                     # Drop rows with ANY missing
df.dropna(how="all")            # Drop rows where ALL are missing
df.dropna(subset=["name"])      # Only check specific columns
df.dropna(thresh=3)             # Keep rows with at least 3 non-null

# Filling missing values
df.fillna(0)                    # Fill all with 0
df["age"].fillna(df["age"].mean())  # Fill with column mean
df["age"].fillna(df["age"].median())  # Fill with median (better for skewed)
df.fillna(method="ffill")       # Forward fill (use previous value)
df.fillna(method="bfill")       # Backward fill

# Interpolation (for time series)
df["value"].interpolate()       # Linear interpolation
df["value"].interpolate(method="time")  # Time-based

# Replace specific values
df.replace({"N/A": None, "": None})
df["status"].replace({"active": 1, "inactive": 0})
```

---

### 17. GroupBy Advanced

```python
# Basic pattern: split-apply-combine
df.groupby("category")["value"].sum()

# Multiple aggregations on same column
df.groupby("category")["value"].agg(["sum", "mean", "count", "std"])

# Different aggregations for different columns
df.groupby("category").agg({
    "value": "sum",
    "quantity": "mean",
    "order_id": "count"
})

# Named aggregations (cleaner output)
df.groupby("category").agg(
    total_value=("value", "sum"),
    avg_quantity=("quantity", "mean"),
    order_count=("order_id", "count")
)

# Apply custom function
def range_func(x):
    return x.max() - x.min()

df.groupby("category")["value"].apply(range_func)

# Transform (returns same shape as original)
df["category_mean"] = df.groupby("category")["value"].transform("mean")

# Filter groups
df.groupby("category").filter(lambda x: x["value"].sum() > 1000)

# Multiple group by columns
df.groupby(["region", "category"])["sales"].sum().reset_index()
```

---

### 18. Pivot Tables & Reshaping

```python
# Pivot table (like Excel)
pd.pivot_table(
    df,
    values="sales",           # What to aggregate
    index="region",           # Rows
    columns="product",        # Columns
    aggfunc="sum",            # How to aggregate
    fill_value=0              # Fill missing with 0
)

# Multiple aggregations
pd.pivot_table(
    df,
    values="sales",
    index="region",
    columns="product",
    aggfunc=["sum", "mean", "count"]
)

# Melt (opposite of pivot - wide to long)
df_long = pd.melt(
    df,
    id_vars=["name"],             # Keep these as-is
    value_vars=["score_1", "score_2"],  # Unpivot these
    var_name="test",              # Name for the variable column
    value_name="score"            # Name for the value column
)

# Stack and unstack
df.stack()        # Columns to rows
df.unstack()      # Rows to columns
```

---

### 19. Merging & Joining

```python
# Merge (like SQL JOIN)
df = pd.merge(
    df1,
    df2,
    on="key",                    # Column to join on
    how="inner"                  # inner, left, right, outer
)

# Different column names
df = pd.merge(df1, df2, left_on="id", right_on="user_id")

# Multiple keys
df = pd.merge(df1, df2, on=["key1", "key2"])

# Concatenate (stack DataFrames)
df = pd.concat([df1, df2])                    # Stack vertically
df = pd.concat([df1, df2], ignore_index=True) # Reset index
df = pd.concat([df1, df2], axis=1)            # Stack horizontally

# Join (merge on index)
df = df1.join(df2, how="left")
```

---

### 20. Window Functions (Rolling & Expanding)

```python
# Rolling window (moving average, etc.)
df["rolling_mean"] = df["value"].rolling(window=7).mean()
df["rolling_sum"] = df["value"].rolling(window=7).sum()
df["rolling_std"] = df["value"].rolling(window=7).std()

# Min periods (handle start of series)
df["rolling_mean"] = df["value"].rolling(window=7, min_periods=1).mean()

# Expanding window (cumulative)
df["cumsum"] = df["value"].expanding().sum()
df["cummax"] = df["value"].expanding().max()

# Shift (lag/lead)
df["prev_value"] = df["value"].shift(1)     # Previous row
df["next_value"] = df["value"].shift(-1)    # Next row

# Percent change
df["pct_change"] = df["value"].pct_change()

# Rank
df["rank"] = df["value"].rank()
df["rank"] = df["value"].rank(ascending=False)

# Difference from previous
df["diff"] = df["value"].diff()
```

---

### 21. Common Interview Patterns

**Pattern 1: Calculate running total per group**
```python
df["running_total"] = df.groupby("category")["value"].cumsum()
```

**Pattern 2: Find top N per group**
```python
df.groupby("category").apply(
    lambda x: x.nlargest(3, "value")
).reset_index(drop=True)
```

**Pattern 3: Flag first/last in group**
```python
df["is_first"] = ~df.duplicated("customer_id")
df["is_last"] = ~df.duplicated("customer_id", keep="last")
```

**Pattern 4: Calculate percentage within group**
```python
df["pct_of_group"] = df["value"] / df.groupby("category")["value"].transform("sum")
```

**Pattern 5: Fill missing with group mean**
```python
df["value"] = df.groupby("category")["value"].transform(
    lambda x: x.fillna(x.mean())
)
```

**Pattern 6: Find consecutive duplicates**
```python
df["is_consecutive_dup"] = df["value"] == df["value"].shift(1)
```

---

### 22. Performance Tips

```python
# Use vectorized operations (FAST)
df["total"] = df["price"] * df["quantity"]  # Good

# Avoid apply with lambdas when possible (SLOW)
df["total"] = df.apply(lambda row: row["price"] * row["quantity"], axis=1)  # Bad

# Use query for filtering (cleaner, sometimes faster)
df.query("age > 25 and status == 'active'")

# Use categories for low-cardinality strings
df["status"] = df["status"].astype("category")

# Read only needed columns
df = pd.read_csv("big_file.csv", usecols=["name", "age"])

# Use chunking for huge files
for chunk in pd.read_csv("huge.csv", chunksize=10000):
    process(chunk)
```

---

## Sample Data Available

Practice with the sample files in `sample_data/`:
- `users.csv` - User data with status filtering
- `sales.csv` - Sales data for groupby and pivot tables
- `products.csv` - Product catalog
- `employees.json` - JSON data to load

```python
import pandas as pd

# Load sample data
users = pd.read_csv("sample_data/users.csv")
sales = pd.read_csv("sample_data/sales.csv")

# Try these exercises:
# 1. Filter active users over age 30
# 2. Calculate total sales by region
# 3. Find top 3 salespersons by total revenue
# 4. Create a pivot table: region vs product
```

---

## Now Try the Problems

1. `p1_numpy_operations.py` - normalize, dot product, cosine similarity, softmax, broadcasting
2. `p2_pandas_wrangling.py` - groupby, rolling average, pivot tables, data cleaning

```bash
pytest 02_numpy_pandas/ -v
```
