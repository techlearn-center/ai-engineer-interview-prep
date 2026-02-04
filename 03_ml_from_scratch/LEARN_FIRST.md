# Learn: ML Algorithms from Scratch

Read this BEFORE attempting the problems.

---

## 1. The Big Picture

All supervised ML follows the same pattern:

```
Training Data (X, y)
       |
       v
[1. Initialize random weights]
       |
       v
[2. Make predictions: y_pred = f(X, weights)]
       |
       v
[3. Measure error: loss = how_wrong(y, y_pred)]
       |
       v
[4. Compute gradients: which direction reduces the error?]
       |
       v
[5. Update weights: move a small step in that direction]
       |
       v
[Repeat steps 2-5 many times (epochs)]
```

This loop is called **gradient descent**.

---

## 2. Linear Regression

**Goal:** Predict a continuous number (e.g., house price, temperature).

**The model:**
```
y_pred = X @ weights + bias
```
That's it. It's a weighted sum of the input features plus a bias term.

**Example:**
```
Features: [square_feet, bedrooms, age]
Weights:  [200,         50000,    -1000]
Bias:     10000

Prediction = 200 * square_feet + 50000 * bedrooms + (-1000) * age + 10000
```

**The loss (how wrong are we?):**
```
MSE = (1/n) * sum((y_true - y_pred)^2)
```
Mean Squared Error - average of squared differences.

**The gradients (how to improve):**
```python
error = y_pred - y          # How much we're off by
dw = (2/n) * X.T @ error   # Gradient for weights
db = (2/n) * sum(error)     # Gradient for bias
```

**The update:**
```python
weights = weights - learning_rate * dw
bias = bias - learning_rate * db
```

**Why subtract?** The gradient points UPHILL (toward more error).
We want to go DOWNHILL (toward less error). So we subtract.

**Learning rate:** How big of a step we take. Too big = overshoot. Too small = slow.

### Try this mental model:
Imagine you're blindfolded on a hilly landscape. You want to find the lowest point.
- You feel the slope under your feet (gradient)
- You take a step downhill (update)
- Repeat until you're at the bottom (convergence)

---

## 3. Logistic Regression

**Goal:** Predict a category (0 or 1). Example: spam or not spam.

**Key difference from linear regression:** Add a sigmoid function to squish
the output to a 0-1 range (probability).

**The sigmoid function:**
```
sigmoid(z) = 1 / (1 + exp(-z))
```

```
Input z:    -inf ... -2 ... 0 ... 2 ... +inf
Output:      0   ... 0.12. 0.5. 0.88..  1
```

**The model:**
```python
z = X @ weights + bias       # Same as linear regression
y_pred = sigmoid(z)           # Squish to 0-1 range
```

**Making a decision:**
```python
prediction = 1 if y_pred >= 0.5 else 0
```

**The gradients (surprisingly same form!):**
```python
error = y_pred - y
dw = (1/n) * X.T @ error
db = (1/n) * sum(error)
```

**Numerical stability tip:**
`exp(-1000)` is fine, but `exp(1000)` overflows to infinity.
Always clip your input: `z = np.clip(z, -500, 500)`

---

## 4. K-Means Clustering

**Goal:** Group similar data points together WITHOUT labels (unsupervised).

**The algorithm:**
```
1. Pick k random points as starting "centroids" (cluster centers)
2. ASSIGN: each data point goes to its nearest centroid
3. UPDATE: move each centroid to the average of its assigned points
4. Repeat 2-3 until nothing changes
```

**Visual example (k=2):**
```
Step 0: Random centroids        Step 1: Assign to nearest
    *                               *
  . . .  .                       o o o  o
     .  .                          o  o

  .  . .                         x  x x
    .                              x
    *                               *

Step 2: Move centroids          Step 3: Reassign (converged!)
    * (moved to center of o's)      *
  o o o  o                       o o o  o
     o  o                          o  o

  x  x x                        x  x x
    x                              x
    * (moved to center of x's)      *
```

**Computing distances (vectorized):**
The trick is to use NumPy broadcasting:
```python
# X shape: (100, 2)  - 100 points, 2 dimensions
# centroids shape: (3, 2)  - 3 centroids, 2 dimensions

X[:, None, :]          # shape (100, 1, 2) - add axis for broadcasting
centroids[None, :, :]  # shape (1, 3, 2)   - add axis for broadcasting

diff = X[:, None, :] - centroids[None, :, :]  # shape (100, 3, 2)
distances = np.sqrt((diff ** 2).sum(axis=2))    # shape (100, 3)
labels = np.argmin(distances, axis=1)            # shape (100,) - nearest centroid
```

**Inertia:** Sum of squared distances from each point to its centroid.
Lower inertia = tighter clusters = better fit.

---

## 5. Key Concepts to Know for Interviews

### Bias vs Variance
- **Underfitting (high bias):** Model too simple, misses patterns
- **Overfitting (high variance):** Model memorizes training data, fails on new data
- **Sweet spot:** Just enough complexity

### Train/Test Split
- Never evaluate on training data
- Split data: 80% train, 20% test
- Train on train set, evaluate on test set

### Feature Scaling
- Gradient descent works better when features are on similar scales
- Z-score: `(x - mean) / std` -> mean=0, std=1
- Min-Max: `(x - min) / (max - min)` -> range [0, 1]

### Learning Rate
- Too high: loss oscillates or diverges
- Too low: training is very slow
- Common values: 0.01, 0.001, 0.0001

---

## Now Try the Problems

Start with:
1. `p1_linear_regression.py` - Implement the full training loop
2. `p2_logistic_regression.py` - Add sigmoid activation
3. `p3_kmeans.py` - Implement the assign-update loop

```bash
pytest 03_ml_from_scratch/ -v
```
