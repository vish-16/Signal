
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
import json # Import json for saving/loading trades
from google.colab import drive # Import google drive - NEW

# ======================
# CONFIGURATION
# ======================
SYMBOLS = ["BTC-USD"]  # List of symbols to monitor - NOW MONITORING BTC-USD
INTERVAL = "15m"
REFRESH_SECONDS = 60
SUPPORT_RESISTANCE_WINDOW = 5
RISK_REWARD_RATIO = 2

EMA_PERIOD = 50 # ADDED EMA 50
WMA_PERIOD = 150 # ADDED WMA 150
STOP_LOSS_BUFFER_BULL = 0.990 # Buffer for Bull Stop Loss - further below EMA
STOP_LOSS_BUFFER_BEAR = 1.010 # Buffer for Bear Stop Loss - further above EMA

# --- GOOGLE DRIVE INTEGRATION ---
drive.mount('/content/drive') # Mount google drive - NEW
TRADES_FILE = "/content/drive/MyDrive/trades.json" # Google Drive path for trades file - NEW

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
                print(f"Calculating EMA for {symbol}...") # Debugging step
                sys.stdout.flush()
                df['EMA_50'] = df['Close'].ewm(span=EMA_PERIOD, adjust=False).mean() # ADDED EMA 50
                print(f"Calculating WMA for {symbol}...") # Debugging step
                sys.stdout.flush()
                df['WMA_150'] = calculate_wma(df['Close'], WMA_PERIOD) # ADDED WMA 150
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

def calculate_wma(series, period):
    weights = np.arange(1, period + 1)
    def wma(x):
        return np.average(x, weights=weights)
    return series.rolling(period).apply(wma, raw=False) # Corrected to use rolling().apply() for WMA


# ======================
# TRADING LOGIC
# ======================
def generate_signals(indicator_data, all_signals_dict, trades_dict): # Pass trades_dict
    all_new_signals = {}
    for symbol, df in indicator_data.items():
        if df.empty:
            print(f"Warning: No indicator data for symbol {symbol} to generate signals.")
            all_new_signals[symbol] = []
            continue

        # --- Ensure 'signal', 'sl_price', 'tp_price' columns exist even if no new signal ---
        df['signal'] = 0  # Initialize signal to 0 (no signal)
        df['sl_price'] = np.nan
        df['tp_price'] = np.nan


        if symbol in trades_dict and trades_dict[symbol] and trades_dict[symbol]['status'] == 'active': # Check for active trade - MODIFIED
            if DEBUG_MODE:
                print(f"--- Active trade detected for {symbol}, skipping signal generation. ---") # Debugging print
            all_new_signals[symbol] = [] # No new signals if trade active
            continue # Skip signal generation if trade is active


        # --- No longer need to initialize df['signal'], df['sl_price'], df['tp_price'] here ---
        # df['signal'] = 0
        # df['sl_price'] = np.nan
        # df['tp_price'] = np.nan
        new_signals = []

        # --- Bullish Signals (EMA 50 > WMA 150) ---
        bull_cond = (df['EMA_50'] > df['WMA_150'])

        # --- Bearish Signals (EMA 50 < WMA 150) ---
        bear_cond = (df['EMA_50'] < df['WMA_150'])


        if DEBUG_MODE:
            print(f"\n--- Bull Signal Check for {symbol} at {df.index[-1]} ---")
            print(f"EMA_50 > WMA 150: {(df['EMA_50'] > df['WMA_150']).iloc[-1]}, EMA_50 Value: {df['EMA_50'].iloc[-1]:.2f}, WMA_150 Value: {df['WMA_150'].iloc[-1]:.2f}")


        if bull_cond.iloc[-1]: # Check for Bull signal
            current_price = df['Close'].iloc[-1]
            sl_price = df['EMA_50'].iloc[-1] * STOP_LOSS_BUFFER_BULL  # SL below EMA 50
            tp_price = df['resistance'].iloc[-1] # TP at Resistance
            df.loc[bull_cond, 'signal'] = 1
            df.loc[bull_cond, 'sl_price'] = sl_price
            df.loc[bull_cond, 'tp_price'] = tp_price
            buy_signals_indices = df[bull_cond].index
            new_signals.extend([(index, 1) for index in buy_signals_indices])
            if DEBUG_MODE:
                print(f"🔥🔥🔥 BULL SIGNAL generated for {symbol} at {df.index[-1]}! 🔥🔥🔥 (EMA 50 > WMA 150)")
            else:
                print(f"🔥🔥🔥 BULL SIGNAL generated for {symbol} at {df.index[-1]}! 🔥🔥🔥")
            trades_dict[symbol] = { # Store trade details - MODIFIED
                'signal_type': 'BUY',
                'entry_price': current_price,
                'sl_price': sl_price,
                'tp_price': tp_price,
                'status': 'active',
                'entry_time': df.index[-1].isoformat() # Capture entry time as ISO string - NEW
            }
            save_trades(trades_dict) # SAVE TRADES AFTER SIGNAL - NEW


        if DEBUG_MODE:
            print(f"\n--- Bear Signal Check for {symbol} at {df.index[-1]} ---")
            print(f"EMA_50 < WMA 150: {(df['EMA_50'] < df['WMA_150']).iloc[-1]}, EMA_50 Value: {df['EMA_50'].iloc[-1]:.2f}, WMA_150 Value: {df['WMA_150'].iloc[-1]:.2f}")


        if bear_cond.iloc[-1]: # Check for Bear signal
            current_price = df['Close'].iloc[-1]
            sl_price = df['EMA_50'].iloc[-1] * STOP_LOSS_BUFFER_BEAR # SL above EMA 50
            tp_price = df['support'].iloc[-1] # TP at Support
            df.loc[bear_cond, 'signal'] = -1
            df.loc[bear_cond, 'sl_price'] = sl_price
            df.loc[bear_cond, 'tp_price'] = tp_price
            sell_signals_indices = df[bear_cond].index
            new_signals.extend([(index, -1) for index in sell_signals_indices])
            if DEBUG_MODE:
                print(f"🔥🔥🔥 BEAR SIGNAL generated for {symbol} at {df.index[-1]}! 🔥🔥🔥 (EMA 50 < WMA 150)")
            else:
                print(f"🔥🔥🔥 BEAR SIGNAL generated for {symbol} at {df.index[-1]}! 🔥🔥🔥")
            trades_dict[symbol] = { # Store trade details - MODIFIED
                'signal_type': 'SELL',
                'entry_price': current_price,
                'sl_price': sl_price,
                'tp_price': tp_price,
                'status': 'active',
                'entry_time': df.index[-1].isoformat() # Capture entry time as ISO string - NEW
            }
            save_trades(trades_dict) # SAVE TRADES AFTER SIGNAL - NEW


        if symbol not in all_signals_dict:
            all_signals_dict[symbol] = []

        all_signals_dict[symbol].extend(new_signals)
        all_new_signals[symbol] = all_new_signals

    return indicator_data, all_signals_dict, all_new_signals, trades_dict # Return trades_dict


def check_trade_closure(indicator_data, trades_dict): # New function to check for trade closure
    print("--- Starting check_trade_closure ---") # Debugging print
    sys.stdout.flush()
    for symbol, trade_info in trades_dict.items():
        if trade_info and trade_info['status'] == 'active': # Check if trade is active
            current_price = indicator_data[symbol]['Close'].iloc[-1]
            signal_type = trade_info['signal_type']
            sl_price = trade_info['sl_price']
            tp_price = trade_info['tp_price']

            if signal_type == 'BUY':
                if current_price <= sl_price: # Check if Stop Loss is hit for Buy trade
                    trades_dict[symbol]['status'] = 'closed_sl'
                    print(f"❌❌❌ STOP LOSS HIT for BUY trade on {symbol} at {current_price:.2f} ❌❌❌")
                    sys.stdout.flush()
                    save_trades(trades_dict) # SAVE TRADES AFTER CLOSURE - NEW
                elif current_price >= tp_price: # Check if Take Profit is hit for Buy trade
                    trades_dict[symbol]['status'] = 'closed_tp'
                    print(f"✅✅✅ TAKE PROFIT HIT for BUY trade on {symbol} at {current_price:.2f} ✅✅✅")
                    sys.stdout.flush()
                    save_trades(trades_dict) # SAVE TRADES AFTER CLOSURE - NEW
            elif signal_type == 'SELL':
                if current_price >= sl_price: # Check if Stop Loss is hit for Sell trade
                    trades_dict[symbol]['status'] = 'closed_sl'
                    print(f"❌❌❌ STOP LOSS HIT for SELL trade on {symbol} at {current_price:.2f} ❌❌❌")
                    sys.stdout.flush()
                    save_trades(trades_dict) # SAVE TRADES AFTER CLOSURE - NEW
                elif current_price <= tp_price: # Check if Take Profit is hit for Sell trade
                    trades_dict[symbol]['status'] = 'closed_tp'
                    print(f"✅✅✅ TAKE PROFIT HIT for SELL trade on {symbol} at {current_price:.2f} ✅✅✅")
                    sys.stdout.flush()
                    save_trades(trades_dict) # SAVE TRADES AFTER CLOSURE - NEW
    print("--- Finished check_trade_closure ---") # Debugging print
    sys.stdout.flush()
    return trades_dict # Return updated trades_dict


# ======================
# PERSISTENCE FUNCTIONS - NEW
# ======================
def save_trades(trades_dict):
    try:
        with open(TRADES_FILE, 'w') as f:
            json.dump(trades_dict, f, indent=4, default=str) # Use default=str for datetime serialization
        print(f"Trades saved to Google Drive: {TRADES_FILE}") # Updated message for Google Drive
        sys.stdout.flush()
    except Exception as e:
        print(f"Error saving trades to Google Drive {TRADES_FILE}: {e}") # Updated message for Google Drive
        sys.stdout.flush()

def load_trades():
    try:
        with open(TRADES_FILE, 'r') as f:
            loaded_trades_dict = json.load(f)
            # Convert entry_time back to datetime object
            for symbol, trade_info in loaded_trades_dict.items():
                if trade_info and 'entry_time' in trade_info:
                    trade_info['entry_time'] = pd.Timestamp(trade_info['entry_time']) # Convert back to Timestamp
            print(f"Trades loaded from Google Drive: {TRADES_FILE}") # Updated message for Google Drive
            sys.stdout.flush()
            return loaded_trades_dict
    except FileNotFoundError:
        print(f"{TRADES_FILE} not found in Google Drive. Starting with empty trades.") # Updated message for Google Drive
        sys.stdout.flush()
        return {} # Return empty dict if file not found
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {TRADES_FILE} in Google Drive. Starting with empty trades.") # Updated message for Google Drive
        sys.stdout.flush()
        return {} # Return empty dict if JSON decode error
    except Exception as e:
        print(f"Error loading trades from Google Drive {TRADES_FILE}: {e}") # Updated message for Google Drive
        sys.stdout.flush()
        return {} # Return empty dict on other errors


# ======================
# VISUALIZATION
# ======================
def plot_data(indicator_data, all_signals_dict, trades_dict): # Pass trades_dict
    if not DEBUG_MODE: # Conditionally clear output only when not in DEBUG_MODE
        clear_output(wait=True)
    num_symbols = len(SYMBOLS)
    for i, symbol in enumerate(SYMBOLS):
        df = indicator_data.get(symbol, pd.DataFrame())
        all_signals = all_signals_dict.get(symbol, [])
        trade_info = trades_dict.get(symbol, None) # Get trade info instead of trades list

        if df.empty:
            print(f"Warning: No data available to plot for symbol {symbol}.")
            continue

        # Convert DataFrame index to UTC for plotting consistency - NEW
        if not df.empty and df.index.tz is not None:
            df.index = df.index.tz_convert('UTC')


        plt.figure(figsize=(20, 10)) # Reduced figure height as subplots are removed

        # --- Subplot 1: Price, EMA 50 and WMA 150 ---
        plt.subplot(2, 1, 1) # Reduced to 2 rows now
        plt.plot(df['Close'], label='Price', alpha=0.7, linewidth=1.5)
        plt.plot(df['EMA_50'], label='EMA 50', color='blue', alpha=0.7) # ADDED EMA 50 Plot
        plt.plot(df['WMA_150'], label='WMA 150', color='orange', alpha=0.7) # ADDED WMA 150 Plot

        # Buy/Sell Signals and SL/TP on Price Chart
        signal_legend_handles = {} # To track legend handles

        # Plotting active trade SL/TP lines from trades_dict - NEW LOGIC
        if trade_info and trade_info['status'] == 'active': # Check for active trade
            entry_time = trade_info['entry_time'] # Get entry time
            sl_price = trade_info['sl_price']
            tp_price = trade_info['tp_price']
            signal_type = trade_info['signal_type']

            color = 'lime' if signal_type == 'BUY' else 'red' # Color for signal markers
            marker = '^' if signal_type == 'BUY' else 'v' # Marker for signal markers
            signal_label = 'Buy Signal' if signal_type == 'BUY' else 'Sell Signal'
            sl_color = 'red' # SL color - RED for both BUY and SELL
            tp_color = 'lime' # TP color - GREEN for both BUY and SELL


            # Convert entry_time to UTC for plotting - NEW
            if isinstance(entry_time, pd.Timestamp) and entry_time.tz is not None:
                entry_time_utc = entry_time.tz_convert('UTC')
            else:
                entry_time_utc = entry_time # Keep as is if already UTC or timezone naive


            # Plot Signal Marker at entry - only once in legend
            if signal_label not in signal_legend_handles:
                signal_legend_handles[signal_label], = plt.plot(entry_time_utc, trade_info['entry_price'], c=color, marker=marker, markersize=10, linestyle='None', markeredgecolor='black', label=signal_label)
            else:
                plt.plot(entry_time_utc, trade_info['entry_price'], c=color, marker=marker, markersize=10, linestyle='None', markeredgecolor='black')


            # Plot SL/TP lines - extending from entry to current time
            if not np.isnan(sl_price):
                plt.hlines(y=sl_price, xmin=entry_time_utc, xmax=df.index[-1], colors=sl_color, linestyles='--', linewidth=2, alpha=0.9, label='Stop Loss' if 'Stop Loss' not in signal_legend_handles else None) # Label SL only once
                plt.plot(entry_time_utc, sl_price, marker='x', markersize=8, color=sl_color, linestyle='None', label='SL Symbol'  if 'Stop Loss' not in signal_legend_handles else None) # SL symbol at entry

            if not np.isnan(tp_price):
                plt.hlines(y=tp_price, xmin=entry_time_utc, xmax=df.index[-1], colors=tp_color, linestyles='-', linewidth=2, alpha=0.9, label='Take Profit' if 'Take Profit' not in signal_legend_handles else None) # Label TP only once
                plt.plot(entry_time_utc, tp_price, marker='o', markersize=8, color=tp_color, linestyle='None', label='TP Symbol' if 'Take Profit' not in signal_legend_handles else None) # TP symbol at entry


        current_price = df['Close'].iloc[-1]
        plt.axhline(current_price, color='blue', linestyle=':', label=f'Current Price: ${current_price:.2f}')
        plt.title(f'{symbol} Trading System - Live Analysis')
        plt.ylabel('Price')
        plt.grid(True)
        plt.legend(handles=list(signal_legend_handles.values())) # Ensure legend handles are in a list


        # --- Subplot 2:  EMA 50 and WMA 150 ---
        plt.subplot(2, 1, 2, sharex=plt.gca()) # Reduced to 2 rows now
        plt.plot(df['EMA_50'], label='EMA 50', color='blue') # ADDED EMA 50 Plot
        plt.plot(df['WMA_150'], label='WMA 150', color='orange') # ADDED WMA 150 Plot
        plt.title('EMA 50 vs WMA 150') # Updated Title
        plt.ylabel('Indicator Value')
        plt.xlabel('Time')
        plt.grid(True)
        plt.legend()


        plt.tight_layout()
        plt.show()



# ======================
# MAIN EXECUTION
# ======================
if __name__ == "__main__":
    # ANSI escape codes for colors
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'
    GRAY_BOLD_ITALICS = '\033[1;3;90m' # ANSI code for Gray, Bold, and Italics
    CYAN_BOLD = '\033[1;96m' # ANSI code for Cyan and Bold - NEW
    RESET_FORMATTING = '\033[0m' # ANSI code to reset formatting


    # Timezone for India
    india_tz = pytz.timezone('Asia/Kolkata')

    # Live Monitoring for Multiple Symbols
    print("\nStarting live monitoring with EMA 50 and WMA 150...") # Updated message
    all_signals_dict = {}
    trades_dict = load_trades() # LOAD TRADES AT STARTUP - NEW
    if not trades_dict: # Initialize if loading failed or no file
        trades_dict = {}
        for symbol in SYMBOLS: # Initialize trade status for each symbol as inactive
            trades_dict[symbol] = None # Initialize trade status to None (no trade yet) - MODIFIED - Initialize to None

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
            trades_dict = check_trade_closure(indicator_data, trades_dict) # Check for trade closure BEFORE signal generation - MODIFIED
            indicator_data, all_signals_dict, new_signals_dict, trades_dict = generate_signals(indicator_data, all_signals_dict, trades_dict) # Pass and get trades_dict
            print("Indicators calculated and signals generated.")
            sys.stdout.flush()


            # Update Display - plot for each symbol
            print("Plotting data...")
            sys.stdout.flush()
            plot_data(indicator_data, all_signals_dict, trades_dict) # Pass trades_dict
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

                    # --- ENSURE 'signal' column exists AGAIN before status update ---
                    df['signal'] = df.get('signal', 0) # Ensure 'signal' column exists, default to 0 if not present
                    # --- End ensure 'signal' column exists ---


                    print(f"\n--- Symbol: {symbol} ---")
                    print(f"  Price: {df['Close'].iloc[-1]:.2f}")
                    print(f"  EMA 50: {df['EMA_50'].iloc[-1]:.2f}") # ADDED EMA 50 Status
                    print(f"  WMA 150: {df['WMA_150'].iloc[-1]:.2f}") # ADDED WMA 150 Status
                    sys.stdout.flush()

                    trade_info = trades_dict.get(symbol) # Get trade info
                    if trade_info and trade_info['status'] == 'active': # Check for active trade status
                        signal_type = trade_info['signal_type'].upper()
                        # --- MODIFIED PRINT STATEMENT FOR GREEN BUY SIGNAL ---
                        if signal_type == 'BUY':
                            print(f"  Current Trade: {GREEN}{signal_type}{RESET}") # Green color for BUY
                        elif signal_type == 'SELL':
                            print(f"  Current Trade: {RED}{signal_type}{RESET}") # Red color for SELL
                        # --- END MODIFIED PRINT STATEMENT ---
                        # --- MODIFIED ENTRY PRICE PRINT STATEMENT ---
                        print(f"    {CYAN_BOLD}Entry Price: {trade_info['entry_price']:.2f}{RESET_FORMATTING}") # Cyan and Bold for Entry Price - NEW
                        # --- END MODIFIED ENTRY PRICE PRINT STATEMENT ---
                        print(f"    Stop Loss (SL): {RED}{trade_info['sl_price']:.2f}{RESET}") # SL in RED
                        print(f"    Take Profit (TP): {GREEN}{trade_info['tp_price']:.2f}{RESET}") # TP in GREEN

                    # --- Debugging check: Is 'signal' column in df? ---
                    if DEBUG_MODE:
                        if 'signal' not in df.columns:
                            print(f"Debug: 'signal' column MISSING from df for symbol {symbol} before signal check!")
                            sys.stdout.flush()
                        else:
                            print(f"Debug: 'signal' column PRESENT in df for symbol {symbol} before signal check.")
                            sys.stdout.flush()
                    # --- End debugging check ---


                    # Check for new signals for this symbol
                    if symbol in all_signals_dict and all_signals_dict[symbol]:
                        last_signal_time, last_signal_type = all_signals_dict[symbol][-1]
                        if last_signal_time == df.index[-1] and 'signal' in df.columns and df['signal'].iloc[-1] != 0: # Added check for 'signal' column existence here too, and wrapped df['signal'].iloc[-1] != 0 in column check
                            signal_type = 'BUY' if last_signal_type == 1 else 'SELL'
                            # --- MODIFIED MESSAGE HERE ---
                            print(f"\n  🔥 TRADE INITIATED: {signal_type} at {df['Close'].iloc[-1]:.2f} 🔥")
                            # --- END MODIFIED MESSAGE HERE ---
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
