# --- CONDA ENVIRONMENT SETUP (Run this section ONCE per Colab session) ---
# Install Conda in Colab environment
!pip install -q condacolab
import condacolab
condacolab.install()

# Verify Conda installation (optional)
!conda --version

# Create a new Conda environment named 'trading_env' with Python 3.9 (or your preferred version)
# and install necessary packages within it.
!conda create -n trading_env python=3.9 yfinance ta-lib pandas matplotlib -c conda-forge

# Activate the Conda environment 'trading_env'
!conda activate trading_env

# --- Rest of your Python script (Run this section every time) ---
import yfinance as yf
import pandas as pd
import talib
import matplotlib.pyplot as plt

# Define the ticker symbol for BTC-USD
ticker = "BTC-USD"

# Fetch historical data
data = yf.download(ticker, period="6mo")

# Calculate Bollinger Bands
close_prices = data['Close'].values.flatten()
upper, middle, lower = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
data['Upper_BB'] = pd.Series(upper, index=data.index)
data['Middle_BB'] = pd.Series(middle, index=data.index)
data['Lower_BB'] = pd.Series(lower, index=data.index)

# Calculate MACD
macd, signal, hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
data['MACD'] = pd.Series(macd, index=data.index)
data['MACD_Signal'] = pd.Series(signal, index=data.index)
data['MACD_Hist'] = pd.Series(hist, index=data.index)

# Remove rows with NaN values
data.dropna(inplace=True)

# Generate Trading Signals - Row-by-row comparison using iterrows() and .item()
buy_signals_list = []
sell_signals_list = []

for index, row in data.iterrows(): # Use iterrows() to iterate through rows
    if (row['Close'].item() < row['Lower_BB'].item()) and (row['MACD'].item() > row['MACD_Signal'].item()): # Access scalar values using .item()
        buy_signals_list.append(True)
    else:
        buy_signals_list.append(False)

    if (row['Close'].item() > row['Upper_BB'].item()) and (row['MACD'].item() < row['MACD_Signal'].item()): # Access scalar values using .item()
        sell_signals_list.append(True)
    else:
        sell_signals_list.append(False)

data['Buy_Signal'] = pd.Series(buy_signals_list, index=data.index)
data['Sell_Signal'] = pd.Series(sell_signals_list, index=data.index)


# Visualize signals on the price chart
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Close Price')
plt.plot(data['Upper_BB'], label='Upper Bollinger Band')
plt.plot(data['Middle_BB'], label='Middle Bollinger Band (SMA)')
plt.plot(data['Lower_BB'], label='Lower Bollinger Band')

# Plot Buy Signals
plt.scatter(data.index[data['Buy_Signal']], data['Close'][data['Buy_Signal']], marker='^', color='green', label='Buy Signal')
# Plot Sell Signals
plt.scatter(data.index[data['Sell_Signal']], data['Close'][data['Sell_Signal']], marker='v', color='red', label='Sell Signal')

plt.title('BTC-USD Price with Bollinger Bands and MACD Signals (Conda Environment)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Print the signals
signals = data[(data['Buy_Signal']) | (data['Sell_Signal'])]
print("\nTrading Signals:")
print(signals[['Close', 'Lower_BB', 'Upper_BB', 'MACD', 'MACD_Signal', 'Buy_Signal', 'Sell_Signal']])
