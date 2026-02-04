# Hints - ML from Scratch

## P1: Linear Regression

### Key Math
```
Prediction:     y_pred = X @ weights + bias
Loss:           MSE = (1/n) * sum((y - y_pred)^2)
Gradient (w):   dw = (2/n) * X.T @ (y_pred - y)
Gradient (b):   db = (2/n) * sum(y_pred - y)
Update:         weights -= lr * dw
                bias -= lr * db
```

### Step by Step
1. Initialize `weights = np.zeros(n_features)` and `bias = 0`
2. Loop for `epochs` iterations:
   - Compute `y_pred = X @ weights + bias`
   - Compute `error = y_pred - y`
   - Compute gradients `dw` and `db` using the formulas above
   - Update weights and bias (subtract lr * gradient)
3. For predict: just compute `X @ weights + bias`

### Common mistakes
- Forgetting to divide by `n_samples` in the gradient
- Using `X @ weights` instead of `X @ self.weights` (using instance vars)

## P2: Logistic Regression

### Key Differences from Linear Regression
- Add sigmoid activation: `sigmoid(z) = 1 / (1 + exp(-z))`
- Output is a probability (0 to 1)
- The gradient formula looks the SAME as linear regression!

### Sigmoid Tips
- Clip input to `[-500, 500]` to avoid overflow: `np.clip(z, -500, 500)`
- sigmoid(0) = 0.5, sigmoid(large positive) ≈ 1, sigmoid(large negative) ≈ 0

### Predict
- `predict_proba`: return sigmoid(X @ weights + bias)
- `predict`: return 1 if predict_proba >= threshold, else 0
- Cast to int: `(probs >= threshold).astype(int)`

## P3: K-Means

### Algorithm
```
1. Pick k random points as initial centroids
2. Repeat:
   a. ASSIGN: each point -> nearest centroid
   b. UPDATE: centroid = mean of assigned points
   c. STOP if assignments didn't change
```

### Assign (vectorized, no loops)
```python
# X has shape (n, d), centroids has shape (k, d)
# Use broadcasting:
X[:, None, :]         # shape (n, 1, d)
centroids[None, :, :] # shape (1, k, d)
diff = X[:, None, :] - centroids[None, :, :]  # shape (n, k, d)
distances = np.sqrt((diff ** 2).sum(axis=2))    # shape (n, k)
labels = np.argmin(distances, axis=1)            # shape (n,)
```

### Update
```python
for i in range(k):
    cluster_points = X[labels == i]
    centroids[i] = cluster_points.mean(axis=0)
```

### Common mistakes
- Not copying centroids during initialization
- Not handling empty clusters (keep old centroid)
- Forgetting to set random seed for reproducibility
