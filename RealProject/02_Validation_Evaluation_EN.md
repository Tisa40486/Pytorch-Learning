# [02] Validation & Evaluation

## 🎯 Objective

Learn how to **measure if your model really works** and avoid common pitfalls like overfitting.

> **Critical Rule**: A model that looks good on training data but fails on new data = useless model.

---

## 📚 Theory

### Why do we need validation?

Imagine you train a model on data and it gets 99% accuracy. Sounds great right?

**BUT** → If you test it on the same data it trained on, of course it memorized it!

It's like studying only the exam answers and expecting to pass a different exam.

### The Solution: Split Your Data

```
Raw Data (100%)
    ↓
[Split]
    ├── Training Set (60-80%) → Model learns from this
    ├── Validation Set (10-20%) → Tune hyperparameters
    └── Test Set (10-20%) → Final evaluation (use ONLY once!)
```

---

## 💻 Part 1: Train/Test Split

### Basic Concept

```python
from sklearn.model_selection import train_test_split

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,                    # Features and target
    test_size=0.2,          # 20% for testing
    random_state=42,        # For reproducibility
    shuffle=True            # Randomize the split
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
```

**Line-by-line explanation:**

1. `test_size=0.2` → 20% goes to test, 80% to train
2. `random_state=42` → Makes it reproducible (same split every time)
3. `shuffle=True` → Randomize before splitting (good for avoiding bias)

### Why random_state=42?

Without it: Every time you run the code, you get a different split.
With random_state: Same split every time = reproducible results = good science.

---

## 💻 Part 2: Time Series Problem - Special Case

**WARNING**: For time series data (like stock prices), DON'T use random shuffle!

Why? Because stock prices have **temporal order**. Shuffling breaks the pattern.

```python
# ❌ WRONG for time series
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True  # ❌ Breaks time order!
)

# ✅ RIGHT for time series
# Take first 80% for training, last 20% for testing
split_point = int(len(data) * 0.8)

X_train = X[:split_point]
X_test = X[split_point:]
y_train = y[:split_point]
y_test = y[split_point:]

# This preserves the time order
```

**Why this matters:**
- Training on 2024-01, 2024-02 → Testing on 2024-03 ✅ (realistic)
- Training on random dates → Testing on random dates ❌ (unrealistic, leaks info)

---

## 💻 Part 3: Evaluation Metrics

After you train a model, you need **metrics** to measure performance.

### For Regression (predicting continuous values like prices)

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Make predictions
y_pred = model.predict(X_test)

# Metric 1: Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: ${mae:.2f}")  # Average absolute error in dollars

# Metric 2: Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.4f}")  # Penalizes large errors more

# Metric 3: Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f"RMSE: ${rmse:.2f}")  # Back to dollar units

# Metric 4: R² Score (coefficient of determination)
r2 = r2_score(y_test, y_pred)
print(f"R²: {r2:.4f}")  # How much variance the model explains (0-1 scale)
```

**Which one to use?**

| Metric | Good For | Range | Interpretation |
|--------|----------|-------|-----------------|
| **MAE** | Intuitive, easy to explain | 0 to ∞ | Average error in original units |
| **RMSE** | Penalizes large errors more | 0 to ∞ | Error in original units |
| **R²** | Overall model quality | 0 to 1 | % of variance explained (higher = better) |

**Example:**
- MAE = $2.5 → Model is wrong by $2.5 on average
- R² = 0.85 → Model explains 85% of the variance in stock price

---

### For Classification (predicting categories)

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Make predictions
y_pred = model.predict(X_test)

# Accuracy: % of correct predictions
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")

# Precision: Of predicted positives, how many were correct?
precision = precision_score(y_test, y_pred)
print(f"Precision: {precision:.2%}")

# Recall: Of actual positives, how many did we catch?
recall = recall_score(y_test, y_pred)
print(f"Recall: {recall:.2%}")

# F1: Balance between precision and recall
f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1:.4f}")
```

---

## 💻 Part 4: Cross-Validation

**Train/test split is good, but has a problem:**

What if your test set is just "lucky" or "unlucky"?

Solution: **Cross-Validation** → Split data into multiple folds and test all of them.

```python
from sklearn.model_selection import cross_val_score

# 5-Fold Cross-Validation
scores = cross_val_score(
    model,                  # Your model
    X, y,                   # Data
    cv=5,                   # 5 splits
    scoring='r2'           # Use R² metric
)

print(f"Fold scores: {scores}")
print(f"Mean score: {scores.mean():.4f}")
print(f"Std dev: {scores.std():.4f}")

# Output example:
# Fold scores: [0.82, 0.85, 0.81, 0.84, 0.83]
# Mean score: 0.8300
# Std dev: 0.0149
```

**How it works:**

```
5-Fold Cross-Validation:

Original data [1,2,3,4,5,6,7,8,9,10]

Fold 1: Train on [2,3,4,5,6,7,8,9,10], Test on [1]
Fold 2: Train on [1,3,4,5,6,7,8,9,10], Test on [2]
Fold 3: Train on [1,2,4,5,6,7,8,9,10], Test on [3]
Fold 4: Train on [1,2,3,5,6,7,8,9,10], Test on [4]
Fold 5: Train on [1,2,3,4,6,7,8,9,10], Test on [5]

Average the 5 results → More reliable estimate!
```

**Why it's better:**
- Each sample is used for both training AND testing
- Reduces randomness from one lucky/unlucky split
- Standard deviation tells you model consistency

---

## 💻 Part 5: Overfitting vs Underfitting

### What is Overfitting?

**Overfitting** = Model memorizes training data instead of learning patterns.

```
Training accuracy: 98%
Test accuracy:     55%  ← HUGE GAP = Overfitting!
```

### What is Underfitting?

**Underfitting** = Model is too simple to learn the pattern.

```
Training accuracy: 60%
Test accuracy:     58%  ← Both low = Underfitting
```

### The Sweet Spot

```
Training accuracy: 85%
Test accuracy:     84%  ← Similar = Good model!
```

### Visualization

```
             Model Complexity
         Low        ↑        High
        
Error  ╱  Underfitting │ Good Fit │ Overfitting  ╲
        Bias too high  │  Sweet   │ Variance too high
                       │  Spot    │
        
Training error → (always goes down)
Test error ─────→ (goes down, then up = overfitting!)
```

---

## 💻 Part 6: How to Detect Overfitting

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='r2'
)

# Calculate means
train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training score', marker='o')
plt.plot(train_sizes, val_mean, label='Validation score', marker='s')
plt.xlabel('Training Set Size')
plt.ylabel('R² Score')
plt.title('Learning Curve - Detect Overfitting')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Interpretation:
# If lines converge → Good fit
# If gap widens → Overfitting
# If both low → Underfitting
```

---

## 🎯 Summary

| Concept | What | Why |
|---------|------|-----|
| **Train/Test Split** | 80/20 split | Evaluate on unseen data |
| **Cross-Validation** | Multiple folds | More reliable estimate |
| **MAE/RMSE** | Error metrics | How wrong the model is |
| **R²** | Quality metric | How much variance explained |
| **Overfitting** | High train, low test | Model memorized |
| **Learning Curve** | Train vs test error | Detect overfitting early |

---

## 🎓 Mini-Exercise

With your `trading_data_clean.csv`:

1. Load the data
2. Split 80/20 (preserve time order for time series!)
3. Train a simple Linear Regression model
4. Calculate MAE, RMSE, R² on test set
5. Create a learning curve

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load your clean data
df = pd.read_csv('trading_data_clean.csv')

# Prepare X (features) and y (target)
X = df[['open', 'high', 'low', 'volume']].values
y = df['close'].values

# YOUR CODE HERE
# 1. Split
# 2. Train model
# 3. Evaluate

# Expected output:
# MAE: $X.XX
# R²: X.XXX
```

---

## 📖 Next Chapter

→ **[03] Train/Test Split & Cross-Validation** : Deep dive into splitting strategies

Let me know when you're ready! 🚀
