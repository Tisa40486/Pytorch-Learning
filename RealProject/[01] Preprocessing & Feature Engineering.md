# [01] Preprocessing & Feature Engineering

## 🎯 Objective

Learn to **prepare data correctly** before training a model.

> **Golden Rule of ML** : 80% of time = cleaning + preparation. 20% = models.

---

## 📚 Theory

### What is preprocessing?

It's transforming your **raw data** into **clean and ready data** for ML.

**Why is it crucial?**
- Models are dumb: they learn what we give them
- Bad data = bad model (even with a good algorithm)
- Preprocessing errors = impossible to debug later

### Main steps

```
Raw data
    ↓
[1] Cleaning
    ↓
[2] Handle missing values
    ↓
[3] Detect outliers
    ↓
[4] Normalization / Standardization
    ↓
[5] Feature Engineering
    ↓
Data ready for ML
```

---

## 💻 Part 1: Basic Cleaning

### Example: Fictional stock dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Create raw data with problems
data = {
    'Date': ['2024-01-01', '2024-01-02', '2024-01-03', 
             '2024-01-04', '2024-01-05', '2024-01-06'],
    'Price': [100.5, 102.3, np.nan, 105.2, 999, 103.5],  # np.nan = missing, 999 = error
    'Volume': [1000, 1200, 1150, np.nan, 1300, 1250],
    'Type': ['BUY', 'SELL', 'BUY', 'SELL', 'INVALID', 'BUY']  # 'INVALID' = error
}

df = pd.DataFrame(data)
print("Raw data:")
print(df)
print("\n")

# ✅ STEP 1: Examine the data
print("=" * 50)
print("STEP 1: Inspect the data")
print("=" * 50)

# See data types
print("Data types:")
print(df.dtypes)
print("\n")

# See missing values
print("Missing values:")
print(df.isnull())
print("\n")

# See statistics
print("Statistics:")
print(df.describe())
```

**Line-by-line explanation:**

1. `import pandas as pd` → Library for data manipulation (dataframes)
2. `import numpy as np` → Math library (np.nan = "no value")
3. `data = {...}` → Create dictionary with messy raw data
4. `df = pd.DataFrame(data)` → Transform into table (dataframe)
5. `df.isnull()` → Find missing values
6. `df.describe()` → Give quick stats (min, max, mean, etc.)

---

## 💻 Part 2: Handle Missing Values

```python
print("\n" + "=" * 50)
print("STEP 2: Handle missing values")
print("=" * 50)

# ❌ Option 1: Drop rows with NaN (SIMPLE but LOSES DATA)
df_dropped = df.dropna()
print("After dropping NaN:")
print(df_dropped)
print(f"We lost {len(df) - len(df_dropped)} rows!\n")

# ✅ Option 2: Fill with mean (BETTER for time series)
df_filled_mean = df.copy()
df_filled_mean['Price'].fillna(df_filled_mean['Price'].mean(), inplace=True)
df_filled_mean['Volume'].fillna(df_filled_mean['Volume'].mean(), inplace=True)
print("After filling with mean:")
print(df_filled_mean)
print("\n")

# ✅ Option 3: Fill with previous value (BETTER for time series)
df_filled_forward = df.copy()
df_filled_forward['Price'].fillna(method='ffill', inplace=True)
df_filled_forward['Volume'].fillna(method='ffill', inplace=True)
print("After forward fill (previous value):")
print(df_filled_forward)
```

**Explanation:**

- `dropna()` → Drop rows with NaN (fast but loses data)
- `fillna(value)` → Fill NaN with a value
- `fillna(method='ffill')` → Fill with last known value (good for time series)
- `inplace=True` → Modify dataframe directly

---

## 💻 Part 3: Detect Outliers

```python
print("\n" + "=" * 50)
print("STEP 3: Detect and remove outliers")
print("=" * 50)

# Using df_filled_forward from previous step
print("Price before outlier removal:")
print(df_filled_forward['Price'].values)

# Method 1: Statistical approach (IQR - Interquartile Range)
Q1 = df_filled_forward['Price'].quantile(0.25)  # 25th percentile
Q3 = df_filled_forward['Price'].quantile(0.75)  # 75th percentile
IQR = Q3 - Q1  # Distance between 25% and 75%

# Outliers are values > Q3 + 1.5*IQR or < Q1 - 1.5*IQR
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Acceptable limits: [{lower_bound:.2f}, {upper_bound:.2f}]")

# Create mask of valid values
mask = (df_filled_forward['Price'] >= lower_bound) & (df_filled_forward['Price'] <= upper_bound)

print(f"Outliers detected: {df_filled_forward[~mask]}")
print("\n")

# Remove outliers
df_clean = df_filled_forward[mask].copy()
print("After removing outliers:")
print(df_clean)
```

**Explanation:**

- `Q1` and `Q3` → 25% and 75% of data (quartiles)
- `IQR` → Distance between Q1 and Q3
- Outlier = value > Q3 + 1.5*IQR (or < Q1 - 1.5*IQR)
- `mask` → Boolean true/false for each row (valid or not)
- `df[mask]` → Keep only rows that are true

---

## 💻 Part 4: Normalization & Standardization

```python
print("\n" + "=" * 50)
print("STEP 4: Normalization & Standardization")
print("=" * 50)

# Before
print("Before normalization:")
print(f"Price min: {df_clean['Price'].min()}, max: {df_clean['Price'].max()}")
print(f"Volume min: {df_clean['Volume'].min()}, max: {df_clean['Volume'].max()}")
print("\n")

# ❌ Problem: Price and Volume are at different scales
# 💡 Solution: Normalize to put them at same scale

# Method 1: NORMALIZATION (Min-Max) → Values between 0 and 1
df_normalized = df_clean.copy()
df_normalized['Price'] = (df_clean['Price'] - df_clean['Price'].min()) / \
                        (df_clean['Price'].max() - df_clean['Price'].min())
df_normalized['Volume'] = (df_clean['Volume'] - df_clean['Volume'].min()) / \
                          (df_clean['Volume'].max() - df_clean['Volume'].min())

print("After normalization (0 to 1):")
print(df_normalized)
print("\n")

# Method 2: STANDARDIZATION (Z-score) → Mean 0, std dev 1
# ✅ Better for most algorithms
df_standardized = df_clean.copy()
scaler = StandardScaler()
df_standardized[['Price', 'Volume']] = scaler.fit_transform(df_clean[['Price', 'Volume']])

print("After standardization (Z-score):")
print(df_standardized)
print(f"Mean Price: {df_standardized['Price'].mean():.6f} (close to 0)")
print(f"Std Dev Price: {df_standardized['Price'].std():.6f} (close to 1)")
```

**Explanation:**

- **Normalization** → Put between 0 and 1: `(x - min) / (max - min)`
- **Standardization** → Mean 0, std 1: `(x - mean) / std_dev`
- `StandardScaler()` → sklearn class that does standardization automatically
- **When to use?** Almost all algorithms (except trees) need it

---

## 💻 Part 5: Feature Engineering

```python
print("\n" + "=" * 50)
print("STEP 5: Feature Engineering")
print("=" * 50)

# Feature Engineering = create new intelligent columns

df_features = df_standardized.copy()
df_features['Date'] = pd.to_datetime(df_clean['Date'])

# Create temporal features
df_features['Day'] = df_features['Date'].dt.day
df_features['Month'] = df_features['Date'].dt.month
df_features['DayOfWeek'] = df_features['Date'].dt.dayofweek  # 0=Monday, 6=Sunday

# Create features based on past values (LAG)
df_features['Price_Previous'] = df_features['Price'].shift(1)  # Previous day value
df_features['Price_Change'] = df_features['Price'] - df_features['Price_Previous']  # Variation
df_features['Price_Change_Pct'] = (df_features['Price_Change'] / df_features['Price_Previous']) * 100

# Create average features (ROLLING)
df_features['Price_MA3'] = df_features['Price'].rolling(window=3).mean()  # 3-day average

print("After Feature Engineering:")
print(df_features)
print("\n")

# Remove NaN created by shift() and rolling()
df_features_clean = df_features.dropna()
print("After removing NaN:")
print(df_features_clean)
```

**Explanation:**

- `shift(1)` → Get value from 1 day before (useful for variations)
- `rolling(window=3).mean()` → Rolling average over 3 days
- `dt.dayofweek` → Day of week (to detect patterns)
- Feature = column we create to help model learn

---

## 📊 Visualize the Result

```python
# Visualize before/after
fig, axes = plt.subplots(2, 1, figsize=(10, 6))

# Before
axes[0].plot(df['Price'], marker='o', label='Raw')
axes[0].set_title("RAW DATA (with NaN and outliers)")
axes[0].set_ylabel("Price")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# After
axes[1].plot(df_features_clean['Price'], marker='o', label='Cleaned', color='green')
axes[1].plot(df_features_clean['Price_MA3'], label='3-day MA', color='red', linestyle='--')
axes[1].set_title("CLEAN DATA (with feature engineering)")
axes[1].set_ylabel("Price")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('preprocessing_comparison.png')
plt.show()

print("✅ Preprocessing complete! Data is ready for ML")
```

---

## 🎯 Summary

| Step | What to do | Why |
|------|-----------|-----|
| **Cleaning** | Remove data entry errors | Valid data |
| **NaN** | Fill or drop | No holes |
| **Outliers** | Detect with IQR | No crazy values |
| **Normalization** | 0-1 or Z-score | Same scale |
| **Features** | Create smart columns | Help the model |

---

## 🎓 Mini-exercise

**Get real CSV data** (https://finance.yahoo.com/ or an API) and:

1. Load data with pandas
2. Find NaN values
3. Apply IQR for outliers
4. Standardize
5. Create a "Change%" column

Test with:
```python
df = pd.read_csv('your_data.csv')
# Your code here
print(df.describe())
```

---

## 📖 Next Chapter

→ **[02] Validation & Evaluation** : How to know if your model works?

Let me know when you're ready! 🚀