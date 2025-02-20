# @title
!pip install yfinance==0.2.54
import yfinance as yf
print(f"yfinance version: {yf.__version__}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from datetime import datetime, timedelta
import time
import pytz  # Import pytz for timezone handling
import sys # Import sys for flushing output

# ======================
# CONFIGURATION
# ======================
SYMBOLS = ["BTC-USD"]  # List of symbols to monitor - NOW MONITORING BTC-USD
INTERVAL = "15m"
REFRESH_SECONDS = 60
TREND_BAND_LEN = 9  # Reduced from 12 - to make trend bands more responsive
TREND_BAND_MULT = 1.5  # Reduced from 2.0 - to make trend bands narrower
MA_FAST_PERIOD = 12
MA_SLOW_PERIOD = 26
RSI_PERIOD = 14
RSI_OVERBOUGHT = 60   # Reduced to 60 - even easier to reach overbought
RSI_OVERSOLD = 40    # Increased to 40 - even easier to reach oversold
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9
SUPPORT_RESISTANCE_WINDOW = 5
RISK_REWARD_RATIO = 2

# --- DEBUG MODE TOGGLE ---
DEBUG_MODE = True  # Set to True to see debugging prints, False to hide them


# ======================
# DATA FUNCTIONS
# ======================
def fetch_data(symbols):
    data = {}
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        try:
            data[symbol] = ticker.history(period="7d", interval=INTERVAL)
        except Exception as e:
            print(f"Error fetching data for {symbol} in full script (simplified fetch_data): {e}")
            data[symbol] = pd.DataFrame()
        return data

# ======================
# TECHNICAL INDICATORS
# ======================
def calculate_indicators(data):
    """Calculate advanced technical indicators for each symbol's DataFrame"""
    indicator_data = {}
    print("--- Starting calculate_indicators ---") # Debugging print at start
    sys.stdout.flush()

    for symbol, df in data.items():
        print(f"Processing symbol: {symbol}") # Debugging symbol
        sys.stdout.flush()
        if df.empty:
            print(f"Warning: DataFrame is empty for symbol {symbol}. Skipping indicator calculation.")
            sys.stdout.flush()
            indicator_data[symbol] = pd.DataFrame()
        else:
            try: # Add try-except WITHIN calculate_indicators
                print(f"Calculating basis for {symbol}...") # Debugging step
                sys.stdout.flush()
                df['basis'] = df['Close'].ewm(span=TREND_BAND_LEN, adjust=False).mean().rolling(TREND_BAND_LEN).mean()
                print(f"Calculating deviation for {symbol}...") # Debugging step
                sys.stdout.flush()
                df['deviation'] = df['Close'].sub(df['basis']).abs().ewm(span=TREND_BAND_LEN*3, adjust=False).mean().mul(TREND_BAND_MULT)
                print(f"Calculating upper band for {symbol}...") # Debugging step - potential error point
                sys.stdout.flush()
                df['upper'] = df['basis'] + df['deviation'] # Potential error point
                print(f"Calculating lower band for {symbol}...") # Debugging step
                sys.stdout.flush()
                df['lower'] = df['basis'] - df['deviation']
                df['trend_band_trend'] = np.where(df['Close'] > df['upper'], 1,
                                                np.where(df['Close'] < df['lower'], -1, 0))
                df['MA_Fast'] = df['Close'].rolling(window=MA_FAST_PERIOD).mean()
                df['MA_Slow'] = df['Close'].rolling(window=MA_SLOW_PERIOD).mean()
                df['RSI'] = calculate_rsi(df['Close'], RSI_PERIOD)
                macd_line, signal_line, hist = calculate_macd(df['Close'], MACD_FAST_PERIOD, MA_SLOW_PERIOD, MACD_SIGNAL_PERIOD)
                df['MACD'] = macd_line
                df['MACD_signal'] = signal_line
                df['MACD_hist'] = hist
                df['resistance'] = df['High'].rolling(SUPPORT_RESISTANCE_WINDOW).max()
                df['support'] = df['Low'].rolling(SUPPORT_RESISTANCE_WINDOW).min()
                indicator_data[symbol] = df
            except Exception as indicator_e: # Catch errors inside indicator calculation
                print(f"Error in calculate_indicators for symbol {symbol}: {indicator_e}") # More specific error message
                sys.stdout.flush()
                indicator_data[symbol] = pd.DataFrame() # Return empty df on error in indicator calc.

    print("--- Finished calculate_indicators ---") # Debugging print at end
    sys.stdout.flush()
    return indicator_data

def calculate_rsi(series, period):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.rolling(window=period, min_periods=period).mean()
    ema_down = down.rolling(window=period, min_periods=period).mean()
    rs = ema_up / ema_down
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast_period, slow_period, signal_period):
    exp1 = series.ewm(span=fast_period, adjust=False).mean()
    exp2 = series.ewm(span=slow_period, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# ======================
# TRADING LOGIC
# ======================
def generate_signals(indicator_data, all_signals_dict):
    all_new_signals = {}
    for symbol, df in indicator_data.items():
        if df.empty:
            print(f"Warning: No indicator data for symbol {symbol} to generate signals.")
            all_new_signals[symbol] = []
            continue

        df['signal'] = 0
        df['sl_price'] = np.nan
        df['tp_price'] = np.nan
        new_signals = []

        # --- Strong Bullish Signals (RSI Confirmed) ---
        strong_bull_cond = (
            (
                (df['MA_Fast'] > df['MA_Slow']) |
                (((df['MA_Slow'] - df['MA_Fast']) < (0.001 * df['Close'])) & (df['MA_Fast'].diff() > 0))
            ) &
            (df['RSI'] < RSI_OVERSOLD) &  # <-- RSI Oversold condition for STRONG signal
            (
                (df['MACD'] > df['MACD_signal']) |
                (df['MACD_hist'] > 0)
            )
        )

        # --- Regular Bullish Signals (No RSI Confirmation) ---
        bull_cond = (
            (
                (df['MA_Fast'] > df['MA_Slow']) |
                (((df['MA_Slow'] - df['MA_Fast']) < (0.001 * df['Close'])) & (df['MA_Fast'].diff() > 0))
            ) &
            #(df['RSI'] < RSI_OVERSOLD) &  # RSI Oversold condition - REMOVED for regular signal
            (
                (df['MACD'] > df['MACD_signal']) |
                (df['MACD_hist'] > 0)
            )
        )

        if DEBUG_MODE:
            print(f"\n--- Bull Signal Check for {symbol} at {df.index[-1]} ---")
            print(f"MA Fast > Slow OR Close to Crossover: {((df['MA_Fast'] > df['MA_Slow']) | (((df['MA_Slow'] - df['MA_Fast']) < (0.001 * df['Close'])) & (df['MA_Fast'].diff() > 0))).iloc[-1]}, MA Values: ...")
            print(f"MACD > Signal OR MACD Hist > 0: {((df['MACD'] > df['MACD_signal']) | (df['MACD_hist'] > 0)).iloc[-1]}, MACD Values: ...")
            print(f"RSI Oversold (for STRONG signal): {(df['RSI'] < RSI_OVERSOLD).iloc[-1]}, RSI Value: {df['RSI'].iloc[-1]=}, Oversold Level: {RSI_OVERSOLD=}") # Debug RSI for STRONG signal check


        if strong_bull_cond.iloc[-1]: # Check for STRONG Bull signal first
            df.loc[strong_bull_cond, 'signal'] = 1
            df.loc[strong_bull_cond, 'sl_price'] = df['support'] * 0.995  # SL at Support (with buffer) - MODIFIED
            df.loc[strong_bull_cond, 'tp_price'] = df['resistance'] # TP at Resistance - MODIFIED
            buy_signals_indices = df[strong_bull_cond].index
            new_signals.extend([(index, 1) for index in buy_signals_indices])
            if DEBUG_MODE:
                print(f"🔥🔥🔥 STRONG BULL SIGNAL generated for {symbol} at {df.index[-1]}! 🔥🔥🔥 (RSI Confirmed)") # Indicate STRONG signal
            else:
                print(f"🔥🔥🔥 BULL SIGNAL generated for {symbol} at {df.index[-1]}! 🔥🔥🔥") # Keep regular signal print for non-debug mode
        elif bull_cond.iloc[-1]: # If STRONG Bull not triggered, check for Regular Bull signal
            df.loc[bull_cond, 'signal'] = 1
            df.loc[bull_cond, 'sl_price'] = df['support'] * 0.995  # SL at Support (with buffer) - MODIFIED
            df.loc[bull_cond, 'tp_price'] = df['resistance'] # TP at Resistance - MODIFIED
            buy_signals_indices = df[bull_cond].index
            new_signals.extend([(index, 1) for index in buy_signals_indices])
            if DEBUG_MODE:
                print(f"🔥🔥🔥 BULL SIGNAL generated for {symbol} at {df.index[-1]}! 🔥🔥🔥 (Regular Signal - No RSI Confirmation)") # Indicate Regular signal
            else:
                print(f"🔥🔥🔥 BULL SIGNAL generated for {symbol} at {df.index[-1]}! 🔥🔥🔥") # Keep regular signal print for non-debug mode


        # --- Strong Bearish Signals (RSI Confirmed) ---
        strong_bear_cond = (
            (
                (df['MA_Fast'] < df['MA_Slow']) |
                (((df['MA_Fast'] - df['MA_Slow']) < (0.001 * df['Close'])) & (df['MA_Fast'].diff() < 0))
            ) &
            (df['RSI'] > RSI_OVERBOUGHT) & # <-- RSI Overbought condition for STRONG signal
            (
                (df['MACD'] < df['MACD_signal']) |
                (df['MACD_hist'] < 0)
            )
        )

        # --- Regular Bearish Signals (No RSI Confirmation) ---
        bear_cond = (
            (
                (df['MA_Fast'] < df['MA_Slow']) |
                (((df['MA_Fast'] - df['MA_Slow']) < (0.001 * df['Close'])) & (df['MA_Fast'].diff() < 0))
            ) &
            #(df['RSI'] > RSI_OVERBOUGHT) & # RSI Overbought condition - REMOVED for regular signal
            (
                (df['MACD'] < df['MACD_signal']) |
                (df['MACD_hist'] < 0)
            )
        )


        if DEBUG_MODE:
            print(f"\n--- Bear Signal Check for {symbol} at {df.index[-1]} ---")
            print(f"MA Fast < Slow OR Close to Crossover: {((df['MA_Fast'] < df['MA_Slow']) | (((df['MA_Fast'] - df['MA_Slow']) < (0.001 * df['Close'])) & (df['MA_Fast'].diff() < 0))).iloc[-1]}, MA Values: ...")
            print(f"MACD < Signal OR MACD Hist < 0: {((df['MACD'] < df['MACD_signal']) | (df['MACD_hist'] < 0)).iloc[-1]}, MACD Values: ...")
            print(f"RSI Overbought (for STRONG signal): {(df['RSI'] > RSI_OVERBOUGHT).iloc[-1]}, RSI Value: {df['RSI'].iloc[-1]=}, Overbought Level: {RSI_OVERBOUGHT=}") # Debug RSI for STRONG signal check


        if strong_bear_cond.iloc[-1]: # Check for STRONG Bear signal first
            df.loc[strong_bear_cond, 'signal'] = -1
            df.loc[strong_bear_cond, 'sl_price'] = df['resistance'] * 1.005 # SL at Resistance (with buffer) - MODIFIED
            df.loc[strong_bear_cond, 'tp_price'] = df['support'] # TP at Support - MODIFIED
            sell_signals_indices = df[strong_bear_cond].index
            new_signals.extend([(index, -1) for index in sell_signals_indices])
            if DEBUG_MODE:
                print(f"🔥🔥🔥 STRONG BEAR SIGNAL generated for {symbol} at {df.index[-1]}! 🔥🔥🔥 (RSI Confirmed)") # Indicate STRONG signal
            else:
                print(f"🔥🔥🔥 BEAR SIGNAL generated for {symbol} at {df.index[-1]}! 🔥🔥🔥") # Keep regular signal print for non-debug mode
        elif bear_cond.iloc[-1]: # If STRONG Bear not triggered, check for Regular Bear signal
            df.loc[bear_cond, 'signal'] = -1
            df.loc[bear_cond, 'sl_price'] = df['resistance'] * 1.005 # SL at Resistance (with buffer) - MODIFIED
            df.loc[bear_cond, 'tp_price'] = df['support'] # TP at Support - MODIFIED
            sell_signals_indices = df[bear_cond].index
            new_signals.extend([(index, -1) for index in sell_signals_indices])
            if DEBUG_MODE:
                print(f"🔥🔥🔥 BEAR SIGNAL generated for {symbol} at {df.index[-1]}! 🔥🔥🔥 (Regular Signal - No RSI Confirmation)") # Indicate Regular signal
            else:
                print(f"🔥🔥🔥 BEAR SIGNAL generated for {symbol} at {df.index[-1]}! 🔥🔥🔥") # Keep regular signal print for non-debug mode


        if symbol not in all_signals_dict:
            all_signals_dict[symbol] = []

        all_signals_dict[symbol].extend(new_signals)
        all_new_signals[symbol] = new_signals

    return indicator_data, all_signals_dict, all_new_signals


# ======================
# VISUALIZATION
# ======================
def plot_data(indicator_data, all_signals_dict, trades_dict):
    if not DEBUG_MODE: # Conditionally clear output only when not in DEBUG_MODE
        clear_output(wait=True)
    num_symbols = len(SYMBOLS)
    for i, symbol in enumerate(SYMBOLS):
        df = indicator_data.get(symbol, pd.DataFrame())
        all_signals = all_signals_dict.get(symbol, [])
        trades = trades_dict.get(symbol, [])

        if df.empty:
            print(f"Warning: No data available to plot for symbol {symbol}.")
            continue

        plt.figure(figsize=(20, 16))

        # --- Subplot 1: Price and Trend Bands ---
        plt.subplot(3, 1, 1)
        plt.plot(df['Close'], label='Price', alpha=0.7, linewidth=1.5)
        plt.plot(df['upper'], label='Upper Band', color='red', alpha=0.3)
        plt.plot(df['lower'], label='Lower Band', color='green', alpha=0.3)

        # Buy/Sell Signals and SL/TP on Price Chart
        signal_legend_handles = {} # To track legend handles
        for signal_time, signal_type in all_signals:
            signal_price = df.loc[signal_time, 'Close']
            sl_price = df.loc[signal_time, 'sl_price']
            tp_price = df.loc[signal_time, 'tp_price']
            color = 'lime' if signal_type == 1 else 'red'
            marker = '^' if signal_type == 1 else 'v'
            signal_label = 'Buy Signal' if signal_type == 1 else 'Sell Signal'


            # Plot Signal Markers
            if signal_label not in signal_legend_handles: # Add label to legend only once
                signal_legend_handles[signal_label], = plt.plot(signal_time, signal_price, c=color, marker=marker, markersize=10, linestyle='None', markeredgecolor='black', label=signal_label)
            else:
                plt.plot(signal_time, signal_price, c=color, marker=marker, markersize=10, linestyle='None', markeredgecolor='black')

            # Plot SL/TP lines - always plot, no legend needed for each SL/TP line
            sl_label = 'Stop Loss' if signal_label not in signal_legend_handles else None # Label SL only once in legend
            tp_label = 'Take Profit' if signal_label not in signal_legend_handles else None # Label TP only once in legend


            plt.hlines(y=sl_price, xmin=signal_time, xmax=df.index[-1], colors=color, linestyles='--', linewidth=1, alpha=0.7, label=sl_label) # SL line
            plt.hlines(y=tp_price, xmin=signal_time, xmax=df.index[-1], colors=color, linestyles='-', linewidth=1, alpha=0.7, label=tp_label)   # TP line


        current_price = df['Close'].iloc[-1]
        plt.axhline(current_price, color='blue', linestyle=':', label=f'Current Price: ${current_price:.2f}')
        plt.title(f'{symbol} Trading System - Live Analysis')
        plt.ylabel('Price')
        plt.grid(True)
        plt.legend(handles=signal_legend_handles.values()) # Use collected handles for signals and SL/TP


        # --- Subplot 2: RSI ---
        plt.subplot(3, 1, 2, sharex=plt.gca())
        plt.plot(df['RSI'], label='RSI', color='purple')
        plt.axhline(RSI_OVERBOUGHT, color='red', linestyle='--', alpha=0.5, label='Overbought')
        plt.axhline(RSI_OVERSOLD, color='green', linestyle='--', alpha=0.5, label='Oversold') # <-- CORRECTED LINE
        plt.title('Relative Strength Index (RSI)')
        plt.ylabel('RSI Value')
        plt.grid(True)
        plt.legend()

        # --- Subplot 3: MACD ---
        plt.subplot(3, 1, 3, sharex=plt.gca())
        plt.plot(df['MACD'], label='MACD', color='blue')
        plt.plot(df['MACD_signal'], label='Signal Line', color='orange')
        plt.bar(df.index, df['MACD_hist'], label='Histogram', color='grey', alpha=0.5)
        plt.axhline(0, color='black', linestyle='-', alpha=0.3)
        plt.title('Moving Average Convergence Divergence (MACD)')
        plt.ylabel('MACD Value')
        plt.xlabel('Time')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()



# ======================
# MAIN EXECUTION
# ======================
if __name__ == "__main__":
    # Timezone for India
    india_tz = pytz.timezone('Asia/Kolkata')

    # Live Monitoring for Multiple Symbols
    print("\nStarting live monitoring for multiple symbols with advanced signals...")
    all_signals_dict = {}
    trades_dict = {}
    while True:
        try:
            print("\n--- Starting new loop iteration ---")
            sys.stdout.flush()

            # Update Data for all symbols
            print("Fetching data...")
            sys.stdout.flush()
            live_data = fetch_data(SYMBOLS)
            print("Data fetched.")
            sys.stdout.flush()

            indicator_data = calculate_indicators(live_data)
            indicator_data, all_signals_dict, new_signals_dict = generate_signals(indicator_data, all_signals_dict)
            print("Indicators calculated and signals generated.")
            sys.stdout.flush()


            # Update Display - plot for each symbol
            print("Plotting data...")
            sys.stdout.flush()
            plot_data(indicator_data, all_signals_dict, trades_dict)
            print("Data plotted.")
            sys.stdout.flush()


            # Get current time in IST and format to 12hr
            current_time_ist = datetime.now(india_tz).strftime('%Y-%m-%d %I:%M:%S %p %Z')

            # Print Status for each symbol
            print(f"\n----- Live Update: {current_time_ist} -----")
            sys.stdout.flush()
            for symbol in SYMBOLS:
                if symbol in indicator_data and not indicator_data[symbol].empty:
                    df = indicator_data[symbol]
                    print(f"\n--- Symbol: {symbol} ---")
                    print(f"  Price: {df['Close'].iloc[-1]:.2f}")
                    trend_text = "Bullish" if df['trend_band_trend'].iloc[-1] == 1 else ("Bearish" if df['trend_band_trend'].iloc[-1] == -1 else "Neutral")
                    print(f"  Trend Band Trend: {trend_text}")
                    print(f"  MA Trend: {'Bullish' if df['MA_Fast'].iloc[-1] > df['MA_Slow'].iloc[-1] else 'Bearish'}")
                    print(f"  RSI: {df['RSI'].iloc[-1]:.2f}")
                    print(f"  MACD Histogram: {df['MACD_hist'].iloc[-1]:.3f}")
                    sys.stdout.flush()


                    # Check for new signals for this symbol
                    if symbol in all_signals_dict and all_signals_dict[symbol]:
                        last_signal_time, last_signal_type = all_signals_dict[symbol][-1]
                        if last_signal_time == df.index[-1] and df['signal'].iloc[-1] != 0:
                            signal_type = 'BUY' if last_signal_type == 1 else 'SELL'
                            print(f"\n  🔥 NEW SIGNAL: {signal_type} at {df['Close'].iloc[-1]:.2f} 🔥")
                            print(f"    Stop Loss (SL): {df['sl_price'].iloc[-1]:.2f}")
                            print(f"    Take Profit (TP): {df['tp_price'].iloc[-1]:.2f}")
                            sys.stdout.flush()
                else:
                    print(f"Warning: No data to display status for symbol {symbol}.")
                    sys.stdout.flush()

            time.sleep(REFRESH_SECONDS)

        except KeyboardInterrupt:
            print("\nMonitoring stopped")
            sys.stdout.flush()
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            sys.stdout.flush()
            time.sleep(30)
