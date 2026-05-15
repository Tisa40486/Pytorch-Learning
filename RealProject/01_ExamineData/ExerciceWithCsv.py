import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('trading_data_messy.csv')

df = pd.DataFrame(data)
print("Raw data:")
print(df)
print("\n")


print(df.dtypes)

df_filled = df.copy()
df_filled['close'] = df_filled['close'].fillna(df_filled['open'].shift(-1))
df_filled['volume'] = df_filled['volume'].fillna(df_filled['volume'].ffill())


Q1 = df_filled['close'].quantile(0.25)
Q3 = df_filled['close'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

mask = (df_filled['close'] >= lower_bound) & (df_filled['close'] <= upper_bound)

df_cleaned = df_filled[mask].copy()

print(df_cleaned)

df_normalized = df_cleaned.copy()
df_normalized['open'] = (df_cleaned['open'] - df_cleaned['open'].min()) / (df_cleaned['open'].max() - df_cleaned['open'].min())
df_normalized['close'] = (df_cleaned['close'] - df_cleaned['close'].min()) / (df_cleaned['close'].max() - df_cleaned['close'].min())
df_normalized['high'] = (df_cleaned['high'] - df_cleaned['high'].min()) / (df_cleaned['high'].max() - df_cleaned['high'].min())
df_normalized['low'] = (df_cleaned['low'] - df_cleaned['low'].min()) / (df_cleaned['low'].max() - df_cleaned['low'].min())
df_normalized['volume'] = (df_cleaned['volume'] - df_cleaned['volume'].min()) / (df_cleaned['volume'].max() - df_cleaned['volume'].min())

def standardize_ticker(group):
    scaler = StandardScaler()
    group[['open', 'close', 'high', 'low', 'volume']] = scaler.fit_transform(
        group[['open', 'close', 'high', 'low', 'volume']]
    )
    return group

df_standardized = df_cleaned.groupby('ticker', group_keys=False).apply(standardize_ticker)

print("\nData after standardization (z-score):")
print(df_standardized)


df_feature = df_cleaned.copy()

df_feature['date'] = pd.to_datetime(df_feature['date'])

df_feature['Day'] = df_feature['date'].dt.day
df_feature['Month'] = df_feature['date'].dt.month
df_feature['Year'] = df_feature['date'].dt.year

df_feature['Price'] = df_feature['high'] - df_feature['low'] 
df_feature['Price_Previous'] =  df_feature['Price'].groupby(df_feature['ticker']).shift(1)  # Create a new column with the previous day's price
df_feature['Price_Change'] = df_feature['Price'] - df_feature['Price_Previous']  # Create a new column for price change 
df_feature['Price_Change_Pct'] = df_feature['Price_Change'] / df_feature['Price_Previous'] * 100  # Create a new column for price change percentage

print("\nData after feature engineering:")
print(df_feature)


tickers = df_feature['ticker'].unique()
n_tickers = len(tickers)

for ticker in tickers:
    ticker_data = df_feature[df_feature['ticker'] == ticker].copy()
    
    ticker_data = ticker_data.sort_values('date')
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    axes[0].plot(ticker_data['date'], ticker_data['high'], marker='o', label='High', color='green', linestyle='--', alpha=0.7)
    axes[0].plot(ticker_data['date'], ticker_data['low'], marker='o', label='Low', color='red', linestyle='--', alpha=0.7)
    axes[0].plot(ticker_data['date'], ticker_data['close'], marker='o', label='Close', color='blue', linewidth=2)
    axes[0].set_title(f"Price for {ticker}", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Price ($)")
    axes[0].set_xlabel("Date")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)
    
    axes[1].bar(ticker_data['date'], ticker_data['volume'], color='black', alpha=0.7)
    axes[1].set_title(f"Volume for {ticker}", fontsize=14, fontweight='bold')
    axes[1].set_ylabel("Volume")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    plt.show()