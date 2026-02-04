# Hints - NumPy & Pandas

## P1: NumPy Operations

### normalize_array
- Formula: `(arr - arr.mean()) / arr.std()`
- Handle edge case: if std == 0, return zeros
- This is also called Z-score normalization or standardization

### batch_dot_product
- Element-wise multiply: `A * B` (NOT `A @ B`)
- Then sum along axis=1: `np.sum(A * B, axis=1)`
- Alternative: `np.einsum('ij,ij->i', A, B)`

### cosine_similarity_matrix
- Step 1: Compute norms per row: `np.linalg.norm(X, axis=1, keepdims=True)`
- Step 2: Normalize: `X_norm = X / norms`
- Step 3: Dot product of normalized matrix: `X_norm @ X_norm.T`
- This gives you the full (n, n) cosine similarity matrix

### softmax
- CRITICAL: Subtract `np.max(x)` first for numerical stability
- Then: `exp_x = np.exp(x - np.max(x))`
- Finally: `exp_x / exp_x.sum()`

### broadcast_operation
- Reshape prices to column: `prices[:, None]` or `prices.reshape(-1, 1)`
- Keep discounts as row: `discounts[None, :]` or just `discounts`
- NumPy broadcasting does the rest: `prices[:, None] * (1 - discounts / 100)`

## P2: Pandas Wrangling

### top_products_by_region
- First: `df.groupby(["region", "product"])["sales"].sum()`
- Then: for each region, get the idxmax
- Or: use `.sort_values().drop_duplicates(subset="region", keep="last")`

### rolling_average_sales
- Sort by date first: `df.sort_values("date")`
- Use: `df["sales"].rolling(window=7, min_periods=1).mean()`
- `min_periods=1` prevents NaN in the first few rows

### pivot_sales_report
- Use `pd.pivot_table(df, values="sales", index="region", columns="product", aggfunc="sum", fill_value=0)`

### clean_messy_data
- `df.drop_duplicates()` - removes exact duplicate rows
- `df["sales"].fillna(df["sales"].median())` - fill NaN with median
- `df["quantity"].clip(lower=0)` - replace negatives with 0
- `df.reset_index(drop=True)` - reset the index
