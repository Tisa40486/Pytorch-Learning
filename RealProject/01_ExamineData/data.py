import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


data = {
    'Date': ['2024-01-01', '2024-01-02', '2024-01-03', 
             '2024-01-04', '2024-01-05', '2024-01-06'],
    'Price': [100.5, 102.3, np.nan, 105.2, 999, 103.5],  
    'Volume': [1000, 1200, 1150, np.nan, 1300, 1250],
    'Type': ['BUY', 'SELL', 'BUY', 'SELL', 'INVALID', 'BUY'] 
}

df = pd.DataFrame(data)

# See Raw data
print("Raw data:")
print(df)
print("\n")

print("=" * 50)
print("STEP 1: Inspect the data")
print("=" * 50)
print("\n")

# See data types
print("Data types:")
print(df.dtypes)
print("\n")

# See missing values
print("Missing values:")
print(df.isnull())
print("\n")

# See Statistics
print("Statistics:")
print(df.describe())
print("\n")


# Handle missing Values

print("\n" + "=" * 50)
print("STEP 2: Handle missing values")
print("=" * 50)
print("\n")

# Option 1: Drop rows with missing values
df_dropped = df.dropna()

print("Data after dropping missing values:")
print(df_dropped)
print("\n")
print(f"We lost {len(df) - len(df_dropped)} rows by dropping missing values.")
print("\n")

# Option 2: Fill missing values with mean
## better if data is not time series, if missing value is random, if missing value is large
df_filled_mean = df.copy()
df_filled_mean['Price'] = df_filled_mean['Price'].fillna(df_filled_mean['Price'].mean(), inplace=True)
df_filled_mean['Volume'] = df_filled_mean['Volume'].fillna(df_filled_mean['Volume'].mean(), inplace=True)

print("Data after filling missing values with mean:")
print(df_filled_mean)
print("\n")

# Option 3: Fill missing values with previous value 
## better if data is time series, if missing value is not random, if missing value is small
df_filled_forward = df.copy()
df_filled_forward['Price'] = df_filled_forward['Price'].ffill()
df_filled_forward['Volume'] = df_filled_forward['Volume'].ffill()
print("Data after filling missing values with forward fill:")
print(df_filled_forward)
print("\n")