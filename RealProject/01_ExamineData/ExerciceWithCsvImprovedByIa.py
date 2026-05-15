"""
IMPROVED VERSION - Your Preprocessing Code
With all best practices applied
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

print("="*70)
print("PREPROCESSING PIPELINE - IMPROVED VERSION")
print("="*70 + "\n")

# ============================================================================
# PART 1: LOAD & INSPECT
# ============================================================================

print("📌 STEP 1: Load and inspect data\n")

# Load data (no need for extra DataFrame conversion)
df = pd.read_csv('trading_data_messy.csv')

print("Raw data:")
print(df.head())
print(f"\nShape: {df.shape}")
print(f"Data types:\n{df.dtypes}\n")
print(f"Missing values:\n{df.isnull().sum()}\n")

# ============================================================================
# PART 2: HANDLE MISSING VALUES
# ============================================================================

print("="*70)
print("📌 STEP 2: Handle missing values\n")

df_filled = df.copy()

# Close price: forward fill, then backward fill if needed
df_filled['close'] = df_filled.groupby('ticker')['close'].fillna(method='ffill')
df_filled['close'] = df_filled.groupby('ticker')['close'].fillna(method='bfill')

# Volume: forward fill
df_filled['volume'] = df_filled.groupby('ticker')['volume'].fillna(method='ffill')

# Open, High, Low: forward fill by ticker
for col in ['open', 'high', 'low']:
    df_filled[col] = df_filled.groupby('ticker')[col].fillna(method='ffill')
    df_filled[col] = df_filled.groupby('ticker')[col].fillna(method='bfill')

print(f"Missing values after filling:")
print(f"{df_filled.isnull().sum().sum()} NaN remaining\n")

# ============================================================================
# PART 3: REMOVE OUTLIERS (per ticker)
# ============================================================================

print("="*70)
print("📌 STEP 3: Remove outliers using IQR\n")

def remove_outliers_iqr(group, column='close', multiplier=1.5):
    """
    Remove outliers for a specific ticker using IQR method
    
    Args:
        group: DataFrame for one ticker
        column: column to detect outliers
        multiplier: IQR multiplier (1.5 standard)
    
    Returns:
        DataFrame with outliers removed
    """
    Q1 = group[column].quantile(0.25)
    Q3 = group[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    # Keep only rows within bounds
    mask = (group[column] >= lower_bound) & (group[column] <= upper_bound)
    
    rows_removed = len(group) - len(group[mask])
    if rows_removed > 0:
        print(f"  {group['ticker'].iloc[0]}: Removed {rows_removed} outliers")
    
    return group[mask]

# Apply to each ticker separately
df_cleaned = df_filled.groupby('ticker', group_keys=False).apply(
    lambda x: remove_outliers_iqr(x, column='close')
)

print(f"\nRows before: {len(df_filled)}, after: {len(df_cleaned)}")
print(f"Total rows removed: {len(df_filled) - len(df_cleaned)}\n")

# ============================================================================
# PART 4: STANDARDIZATION (Z-score) - BETTER THAN NORMALIZATION
# ============================================================================

print("="*70)
print("📌 STEP 4: Standardization (Z-score per ticker)\n")

def standardize_by_ticker(group):
    """
    Standardize numeric columns per ticker
    This prevents information leakage between different stocks
    """
    scaler = StandardScaler()
    numeric_cols = ['open', 'close', 'high', 'low', 'volume']
    
    group[numeric_cols] = scaler.fit_transform(group[numeric_cols])
    return group

df_standardized = df_cleaned.groupby('ticker', group_keys=False).apply(standardize_by_ticker)

print("Standardization complete!")
print(f"Example (first 3 rows of AAPL):")
print(df_standardized[df_standardized['ticker'] == 'AAPL'].head(3))
print(f"\nMean (should be ≈0): {df_standardized[['open', 'close', 'high', 'low', 'volume']].mean()}")
print(f"Std Dev (should be ≈1): {df_standardized[['open', 'close', 'high', 'low', 'volume']].std()}\n")

# ============================================================================
# PART 5: FEATURE ENGINEERING
# ============================================================================

print("="*70)
print("📌 STEP 5: Feature Engineering\n")

df_features = df_standardized.copy()

# Convert date to datetime
df_features['date'] = pd.to_datetime(df_features['date'])

# Temporal features
print("Creating temporal features...")
df_features['day'] = df_features['date'].dt.day
df_features['month'] = df_features['date'].dt.month
df_features['year'] = df_features['date'].dt.year
df_features['dayofweek'] = df_features['date'].dt.dayofweek  # 0=Monday, 6=Sunday

# Price-based features (per ticker)
print("Creating price-based features...")
df_features['intraday_range'] = df_features['high'] - df_features['low']  # Daily volatility

# Previous close (per ticker, not across tickers)
df_features['close_previous'] = df_features.groupby('ticker')['close'].shift(1)

# Price change (absolute and percentage)
df_features['price_change'] = df_features['close'] - df_features['close_previous']
df_features['price_change_pct'] = (df_features['price_change'] / df_features['close_previous'].abs()) * 100

# Moving averages (per ticker)
print("Creating moving averages...")
df_features['close_ma2'] = df_features.groupby('ticker')['close'].transform(
    lambda x: x.rolling(window=2, min_periods=1).mean()
)
df_features['close_ma3'] = df_features.groupby('ticker')['close'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)

# Volume moving average
df_features['volume_ma3'] = df_features.groupby('ticker')['volume'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)

print("\nFeatures created successfully!\n")

# ============================================================================
# PART 6: FINAL CLEAN DATASET
# ============================================================================

print("="*70)
print("📌 STEP 6: Create final dataset\n")

# Remove NaN created by shift() and rolling()
df_final = df_features.dropna()

print(f"Rows before NaN removal: {len(df_features)}")
print(f"Rows after NaN removal: {len(df_final)}")
print(f"Rows removed: {len(df_features) - len(df_final)}")

# Check for remaining issues
print(f"\nRemaining issues:")
print(f"  NaN: {df_final.isnull().sum().sum()}")
print(f"  Duplicates: {df_final.duplicated().sum()}")

print(f"\nFinal dataset info:")
print(f"  Shape: {df_final.shape}")
print(f"  Columns: {df_final.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(df_final.head())

# ============================================================================
# PART 7: VISUALIZATION
# ============================================================================

print("\n" + "="*70)
print("📌 STEP 7: Visualize results per ticker\n")

tickers = df_final['ticker'].unique()

for ticker in tickers:
    ticker_data = df_final[df_final['ticker'] == ticker].copy()
    ticker_data = ticker_data.sort_values('date')
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: High/Low/Close prices
    axes[0].plot(ticker_data['date'], ticker_data['high'], 
                marker='o', label='High', color='green', linestyle='--', alpha=0.7, linewidth=1.5)
    axes[0].plot(ticker_data['date'], ticker_data['low'], 
                marker='o', label='Low', color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    axes[0].plot(ticker_data['date'], ticker_data['close'], 
                marker='o', label='Close', color='blue', linewidth=2)
    axes[0].fill_between(ticker_data['date'], ticker_data['high'], ticker_data['low'], 
                        alpha=0.2, color='gray', label='Daily Range')
    axes[0].set_title(f"📊 Price Data for {ticker}", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Standardized Price")
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Price change percentage
    colors = ['green' if x > 0 else 'red' for x in ticker_data['price_change_pct']]
    axes[1].bar(ticker_data['date'], ticker_data['price_change_pct'], 
               color=colors, alpha=0.7, label='Daily Change %')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_title(f"📈 Daily Price Change % for {ticker}", fontsize=14, fontweight='bold')
    axes[1].set_ylabel("Change %")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Volume
    axes[2].bar(ticker_data['date'], ticker_data['volume'], 
               color='black', alpha=0.7, label='Volume')
    axes[2].plot(ticker_data['date'], ticker_data['volume_ma3'], 
                color='orange', linewidth=2, label='3-day MA')
    axes[2].set_title(f"📦 Volume for {ticker}", fontsize=14, fontweight='bold')
    axes[2].set_ylabel("Standardized Volume")
    axes[2].set_xlabel("Date")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'preprocessing_{ticker}.png', dpi=150, bbox_inches='tight')
    print(f"✅ Graph saved: preprocessing_{ticker}.png")
    plt.show()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("✅ PREPROCESSING COMPLETE - SUMMARY")
print("="*70 + "\n")

summary = f"""
Data Processing Summary:
{'='*60}
Initial rows:         {len(df)}
After cleaning NaN:   {len(df_filled)}
After outlier removal: {len(df_cleaned)}
Final (no NaN):       {len(df_final)}

Features created:     {len([c for c in df_final.columns if '_' in c])}
Total columns:        {len(df_final.columns)}

Ready for ML:         ✅ YES

Next step:            [02] Validation & Evaluation
{'='*60}
"""

print(summary)

# Save final dataset
df_final.to_csv('trading_data_clean.csv', index=False)
print(f"\n💾 Final dataset saved to 'trading_data_clean.csv'")
print(f"\n✨ Data is now ready for model training!\n")