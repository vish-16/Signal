import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Configuration
SYMBOL = 'BTC-USD'
START_DATE = '2020-01-01'
END_DATE = '2023-01-01'

# Fetch historical data
def fetch_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    return data

# Calculate technical indicators
def calculate_indicators(data):
    # Exponential Moving Averages
    data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['EMA200'] = data['Close'].ewm(span=200, adjust=False).mean()

    # MACD
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rs.replace([np.inf, -np.inf], np.nan, inplace=True)
    data['RSI'] = 100 - (100 / (1 + rs))
    data['RSI'].fillna(100, inplace=True)

    # Bollinger Bands
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['STD20'] = data['Close'].rolling(window=20).std()
    data['Upper_Band'] = data['MA20'] + (data['STD20'] * 2)
    data['Lower_Band'] = data['MA20'] - (data['STD20'] * 2)

    return data

# Generate trading signals
def generate_signals(data):
    signals = []
    for i in range(len(data)):
        if i == 0: # No signal on the first day as shift(-1) is not available
            signals.append(0)
            continue

        ema_cross_up = (data['EMA50'].iloc[i] > data['EMA200'].iloc[i]) and (data['EMA50'].iloc[i-1] <= data['EMA200'].iloc[i-1])
        ema_cross_down = (data['EMA50'].iloc[i] < data['EMA200'].iloc[i]) and (data['EMA50'].iloc[i-1] >= data['EMA200'].iloc[i-1])

        macd_cross_up = (data['MACD'].iloc[i] > data['Signal_Line'].iloc[i]) and (data['MACD'].iloc[i-1] <= data['Signal_Line'].iloc[i-1])
        macd_cross_down = (data['MACD'].iloc[i] < data['Signal_Line'].iloc[i]) and (data['MACD'].iloc[i-1] >= data['Signal_Line'].iloc[i-1])

        rsi_oversold = data['RSI'].iloc[i] < 30
        rsi_overbought = data['RSI'].iloc[i] > 70

        price_below_lower = data['Close'].iloc[i] < data['Lower_Band'].iloc[i]
        price_above_upper = data['Close'].iloc[i] > data['Upper_Band'].iloc[i]

        buy_signal = ema_cross_up and macd_cross_up and rsi_oversold and price_below_lower
        sell_signal = ema_cross_down and macd_cross_down and rsi_overbought and price_above_upper

        if buy_signal:
            signals.append(1)
        elif sell_signal:
            signals.append(-1)
        else:
            signals.append(0)

    data['Signal'] = signals
    return data

# Visualize results
def plot_signals(data):
    plt.figure(figsize=(16, 12))

    # Price and indicators
    plt.subplot(3, 1, 1)
    plt.plot(data['Close'], label='Price', color='black')
    plt.plot(data['EMA50'], label='EMA 50', alpha=0.75)
    plt.plot(data['EMA200'], label='EMA 200', alpha=0.75)
    plt.plot(data['Upper_Band'], label='Upper Bollinger', linestyle='--', color='gray')
    plt.plot(data['Lower_Band'], label='Lower Bollinger', linestyle='--', color='gray')
    plt.scatter(data[data['Signal'] == 1].index,
                data[data['Signal'] == 1]['Close'],
                marker='^', color='g', s=100, label='Buy Signal')
    plt.scatter(data[data['Signal'] == -1].index,
                data[data['Signal'] == -1]['Close'],
                marker='v', color='r', s=100, label='Sell Signal')
    plt.title('Price Chart with Trading Signals')
    plt.legend()

    # MACD
    plt.subplot(3, 1, 2)
    plt.plot(data['MACD'], label='MACD', color='blue')
    plt.plot(data['Signal_Line'], label='Signal Line', color='orange')
    plt.title('MACD Indicator')
    plt.legend()

    # RSI
    plt.subplot(3, 1, 3)
    plt.plot(data['RSI'], label='RSI', color='purple')
    plt.axhline(30, linestyle='--', color='green', alpha=0.5)
    plt.axhline(70, linestyle='--', color='red', alpha=0.5)
    plt.title('Relative Strength Index (RSI)')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Fetch and process data
    data = fetch_data(SYMBOL, START_DATE, END_DATE)
    data = calculate_indicators(data)
    data = generate_signals(data)

    # Display signals
    print("Buy Signals:")
    print(data[data['Signal'] == 1][['Close', 'EMA50', 'EMA200', 'RSI']])
    print("\nSell Signals:")
    print(data[data['Signal'] == -1][['Close', 'EMA50', 'EMA200', 'RSI']])

    # Visualize results
    plot_signals(data)
