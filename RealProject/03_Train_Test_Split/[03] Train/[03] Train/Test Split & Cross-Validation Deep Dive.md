# [03] Train/Test Split & Cross-Validation Deep Dive

## 🎯 Objective

Master **different splitting strategies** for different types of data and understand why it matters.

> **Key Principle**: How you split your data affects your model's evaluation. Wrong split = wrong conclusions.

---

## 📚 Theory

### Why Different Splitting Strategies?

Not all data is the same:

- **Random data** (images, text) → Can shuffle randomly
- **Time series data** (stocks, weather) → Must preserve order
- **Imbalanced data** (fraud detection) → Need stratification
- **Limited data** → Need better splitting strategy

---

## 💻 Part 1: Basic Train/Test Split Review

### Standard Approach

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,          # 20% test, 80% train
    random_state=42,        # Reproducibility
    shuffle=True            # Random shuffle
)
```

**When to use**: Image classification, text classification, non-temporal data

**Pros:**
- Simple
- Widely applicable
- Reproducible

**Cons:**
- Loses temporal information (bad for time series)
- Single split = random variation
- Not ideal for imbalanced data

---

## 💻 Part 2: Time Series Split (Critical for Finance!)

### The Problem with Random Split

```
Stock Data [Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct]

❌ Random split:
Training: [Jan, Mar, May, Jul, Sep, Feb, Apr]  ← Mixed dates!
Test: [Jun, Aug, Oct, ...]

This is WRONG because:
- Model trains on future data (Jun, Aug) before seeing past data
- Leaks information from future into past
- Not realistic for trading (you can't trade with future data)
```

### The Right Way: Forward-Looking Split

```
✅ Time Series Split:
Training: [Jan, Feb, Mar, Apr, May]
Test: [Jun, Jul, Aug, Sep, Oct]

This is CORRECT because:
- Model learns from past
- Evaluates on future
- Realistic for trading
```

### Implementation

```python
# Manual time series split (simple)
split_point = int(len(data) * 0.8)
X_train = X[:split_point]
X_test = X[split_point:]
y_train = y[:split_point]
y_test = y[split_point:]

# sklearn TimeSeriesSplit (better for cross-validation)
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # train and evaluate
```

### Visualization of TimeSeriesSplit

```
Data: [1,2,3,4,5,6,7,8,9,10]

Split 1:
  Train: [1,2]     Test: [3]

Split 2:
  Train: [1,2,3]   Test: [4]

Split 3:
  Train: [1,2,3,4] Test: [5]

Split 4:
  Train: [1,2,3,4,5] Test: [6]

Split 5:
  Train: [1,2,3,4,5,6] Test: [7]

Each fold grows! Training set increases, test is always future.
```

---

## 💻 Part 3: Stratified Split (For Imbalanced Data)

### The Problem with Random Split on Imbalanced Data

```
Dataset: 95% class 0, 5% class 1

❌ Unlucky random split:
Training: 94% class 0, 6% class 1  ← Good distribution
Test: 97% class 0, 3% class 1  ← Bad distribution!

Result: Test accuracy is misleading!
```

### Solution: Stratified Split

```python
from sklearn.model_selection import StratifiedKFold

# Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,           # ← Key! Keeps same distribution
    random_state=42
)

# Now both train and test have ~95% class 0, ~5% class 1

# For cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
```

---

## 💻 Part 4: K-Fold Cross-Validation

### Basic K-Fold

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_scores = []

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate
    score = model.score(X_test, y_test)
    fold_scores.append(score)
    
    print(f"Fold score: {score:.4f}")

print(f"\nAverage: {np.mean(fold_scores):.4f}")
print(f"Std Dev: {np.std(fold_scores):.4f}")
```

### Advantages of K-Fold

```
Single Train/Test Split:
- Uses 80% data for training
- 20% wasted (only used for evaluation)
- One evaluation = lucky/unlucky?

5-Fold Cross-Validation:
- Each data point used 4x for training, 1x for testing
- Better use of limited data
- 5 independent evaluations = more reliable
- Std dev shows consistency
```

---

## 💻 Part 5: Leave-One-Out Cross-Validation (LOOCV)

### What It Is

```
LOOCV with 10 samples:

Iteration 1: Train on [2,3,4,5,6,7,8,9,10],  Test on [1]
Iteration 2: Train on [1,3,4,5,6,7,8,9,10],  Test on [2]
Iteration 3: Train on [1,2,4,5,6,7,8,9,10],  Test on [3]
...
Iteration 10: Train on [1,2,3,4,5,6,7,8,9], Test on [10]

Average 10 results → Best estimate, but SLOW!
```

### Implementation

```python
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()

# For scoring
scores = cross_val_score(model, X, y, cv=loo, scoring='r2')
print(f"LOOCV Score: {scores.mean():.4f}")

# WARNING: This is VERY slow for large datasets!
# With 1000 samples = 1000 model trainings
# Only use if you have <200 samples
```

### When to Use

| Method | Pros | Cons | Use When |
|--------|------|------|----------|
| **Train/Test Split** | Fast, simple | Less reliable | Large dataset (>10k) |
| **K-Fold (5-10)** | Good balance | Slower than split | Medium dataset (1k-10k) |
| **Stratified K-Fold** | Handles imbalance | Medium speed | Imbalanced classes |
| **Time Series Split** | Realistic for time series | Special use case | Time series data |
| **LOOCV** | Best estimate | VERY slow | Small dataset (<200) |

---

## 💻 Part 6: Group K-Fold (Multiple Related Samples)

### The Problem

```
You have multiple stocks (AAPL, MSFT, GOOGL)

❌ Normal K-Fold:
Fold 1: Train [AAPL_jan, MSFT_feb, GOOGL_jan], Test [AAPL_feb, MSFT_jan, GOOGL_feb]

Problem: Train and test have data from same company!
Leaks information from company to company

✅ Group K-Fold:
Fold 1: Train on [AAPL], Test on [MSFT, GOOGL]
Fold 2: Train on [MSFT], Test on [AAPL, GOOGL]
Fold 3: Train on [GOOGL], Test on [AAPL, MSFT]

Each company's data stays together!
```

### Implementation

```python
from sklearn.model_selection import GroupKFold

groups = df['ticker'].values  # [AAPL, AAPL, ..., MSFT, MSFT, ..., GOOGL]

gkf = GroupKFold(n_splits=5)

for train_idx, test_idx in gkf.split(X, y, groups=groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Train and evaluate
```

---

## 💻 Part 7: Compare All Strategies

### Visual Comparison

```
Data: 100 samples

Train/Test Split (80/20):
████████████████████████████████ (80 train)
████████ (20 test)


5-Fold Cross-Validation:
Fold 1: ████████████████ | ████
Fold 2: ████ ████████████████ |
Fold 3: | ████████████████ | ████
Fold 4: ████ | ████████████████
Fold 5: ████████████████ | ████

(More data used for training in each fold)
```

### Which Metric to Trust?

```
Train/Test Split:
  R² = 0.85
  
  Problem: Could be lucky/unlucky split
  Confidence: Medium

5-Fold Cross-Validation:
  R² = 0.82 ± 0.04 (scores: [0.80, 0.81, 0.83, 0.84, 0.82])
  
  Better: Know variability
  Confidence: High
```

---

## 🎯 Decision Tree: Which Strategy to Use?

```
Do you have time series data?
  YES → Use TimeSeriesSplit or manual time-based split
  NO  → Continue

Is your data imbalanced (class imbalance)?
  YES → Use StratifiedKFold
  NO  → Continue

Do you have group/cluster structure (multiple companies, etc)?
  YES → Use GroupKFold
  NO  → Continue

How much data do you have?
  < 200 samples → LOOCV (or 10-Fold)
  200-1000 → 5-Fold or 10-Fold
  > 1000 → 5-Fold is fine, or even train/test split
```

---

## 💻 Part 8: Common Mistakes

### Mistake 1: Using Same Data for Selection and Evaluation

```python
# ❌ WRONG
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train 100 models with different hyperparameters
best_model = None
best_score = 0
for param in param_list:
    model = Model(param)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)  # ❌ Using test set for selection!
    if score > best_score:
        best_score = score
        best_model = model
```

**Problem**: Test set is now part of training! You optimized for test set specifically.

```python
# ✅ CORRECT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

# Use validation for selection
best_model = None
best_score = 0
for param in param_list:
    model = Model(param)
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)  # ✅ Validation set
    if score > best_score:
        best_score = score
        best_model = model

# Use test set only for final evaluation
final_score = best_model.score(X_test, y_test)
```

### Mistake 2: Scaling Before Split

```python
# ❌ WRONG
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # ❌ Fits on ALL data

X_train, X_test = train_test_split(X_scaled, test_size=0.2)

# Problem: Test set data was used to compute scaling parameters!
# Information leakage!
```

```python
# ✅ CORRECT
X_train, X_test = train_test_split(X, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # ✅ Fit on train only
X_test_scaled = scaler.transform(X_test)  # Transform test with train's parameters
```

### Mistake 3: Reporting Single Split Metrics

```python
# ❌ POOR
"Our model achieves R² = 0.85"

# Problem: Could be lucky split

# ✅ GOOD
"Our model achieves R² = 0.82 ± 0.04 (5-fold CV)"

# Shows: Average and variability
```

---

## 🎯 Summary

| Strategy | Use Case | Pros | Cons |
|----------|----------|------|------|
| **Train/Test** | Large non-temporal data | Simple, fast | Single estimate |
| **K-Fold** | General purpose | Reliable estimate | Slower |
| **Stratified K-Fold** | Imbalanced data | Maintains distribution | Slightly slower |
| **Time Series Split** | Time series (stocks) | Realistic evaluation | Specific to sequential |
| **Group K-Fold** | Multiple groups (companies) | No info leakage | Complex setup |
| **LOOCV** | Very small data (<200) | Best estimate | VERY slow |

---

## 🎓 Mini-Exercise

With your `trading_data_clean.csv`:

1. **Train/Test Split**: Split 80/20 (preserve time order)
2. **5-Fold CV**: Evaluate with 5-fold cross-validation
3. **Time Series Split**: Use TimeSeriesSplit with 5 splits
4. **Compare results**: Are they similar or different?

```python
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LinearRegression

df = pd.read_csv('trading_data_clean.csv')
X = df[['open', 'high', 'low', 'volume']].values
y = df['close'].values

model = LinearRegression()

# 1. Train/Test Split evaluation
split_point = int(len(X) * 0.8)
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]
model.fit(X_train, y_train)
split_score = model.score(X_test, y_test)
print(f"Train/Test Split R²: {split_score:.4f}")

# 2. K-Fold CV
kf_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"K-Fold R²: {kf_scores.mean():.4f} ± {kf_scores.std():.4f}")

# 3. Time Series Split
tscv = TimeSeriesSplit(n_splits=5)
ts_scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
print(f"Time Series Split R²: {ts_scores.mean():.4f} ± {ts_scores.std():.4f}")

# Compare
print("\nWhich is most reliable for stock prediction?")
print("Answer: Time Series Split (respects temporal order)")
```

---

## 📖 Next Chapter

→ **[04] Decision Trees & Random Forest** : First real ML algorithms that matter

Let me know when you're ready! 🚀