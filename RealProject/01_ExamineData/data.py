import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler


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
df_filled_mean['Price'] = df_filled_mean['Price'].fillna(df_filled_mean['Price'].mean())
df_filled_mean['Volume'] = df_filled_mean['Volume'].fillna(df_filled_mean['Volume'].mean())

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


print("\n" + "=" * 50)
print("STEP 3: Handle outliers")
print("=" * 50)
print("\n")

print("Price before outlier removal:")
print(df_filled_forward['Price'].values)
print("\n")

# Calculate Interquartile Range (IQR) for Price
Q1 = df_filled_forward['Price'].quantile(0.25)
Q3 = df_filled_forward['Price'].quantile(0.75)
IQR = Q3 - Q1

# Bound are like the true limit of the data, if data is outside of this limit, it's considered as outlier
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
#"1.5 * IQR" is a statistic convention, use for a balance between removing outliers and keeping valid data, but it can be adjusted based on the specific dataset and requirements.

print(f"Lower bound for Price: {lower_bound}")
print(f"Upper bound for Price: {upper_bound}")
print("\n")

# Create mask of values within the bounds
## mask is a filter, boolean array that indicate if the value is an outlier or not, True if it's not an outlier, False if it's an outlier
mask = (df_filled_forward['Price'] >= lower_bound) & (df_filled_forward['Price'] <= upper_bound)

print(f"Mask for outlier detection: {mask.values}")
print("\n")
print("Data with valid Price values:")
print(df_filled_forward[mask])


#remove outliers
df_cleaned = df_filled_forward[mask].copy()
print("\nData after removing outliers:")
print(df_cleaned)


print("\n" + "=" * 50)
print("STEP 4: Normalize data")
print("=" * 50)

# Normalization is the process of scaling the data to a specific range, 
# usually [0, 1], to ensure that all features contribute equally to the analysis 
# and to improve the performance of machine learning algorithms.

#Before normalization
df_normalized = df_cleaned.copy()
df_normalized['Price'] = (df_normalized['Price'] - df_normalized['Price'].min()) / (df_normalized['Price'].max() - df_normalized['Price'].min())
#Normalization formula: (x - min) / (max - min) to scale the data to [0, 1]

df_normalized['Volume'] = (df_normalized['Volume'] - df_normalized['Volume'].min()) / (df_normalized['Volume'].max() - df_normalized['Volume'].min())
print("\nData after normalization:")

print(df_normalized)
print("\n")

print("\n" + "=" * 50)
print("STEP 4: Standardize data")
print("=" * 50)

# Standardization is the process of scaling the data 
# to have a mean of 0 and a standard deviation of 1,
# which can help to improve the performance of machine learning algorithms,
# especially those that are sensitive to the scale of the data.
## better for most algorithms

df_standardized = df_cleaned.copy()
scaler = StandardScaler()
#Standardization formula: 
# (x - mean) (= center the data around 0)
# / std (= scale the data to have a standard deviation of 1)
# to scale the data to have mean 0 and std 1

df_standardized[['Price', 'Volume']] = scaler.fit_transform(df_cleaned[['Price', 'Volume']])

print("\nData after standardization (z-score):")
print(df_standardized)
print(f"Mean of Price after standardization: {df_standardized['Price'].mean()}")
print(f"Std Dev Price: {df_standardized['Price'].std():.6f} (close to 1)")