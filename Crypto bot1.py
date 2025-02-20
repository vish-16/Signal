import yfinance as yf
import pandas as pd
import time
from datetime import datetime, timedelta
import pytz  # For timezone handling
import os # For file operations

# --- SETTINGS / PARAMETERS ---
SYMBOL = "BTC-USD"  # Example symbol - you can change this
TIME_INTERVAL = "15m"  # 15-minute intervals - you can change this (e.g., "5m", "1h", "1d")
MOVING_AVERAGE_FAST_PERIOD = 12
MOVING_AVERAGE_SLOW_PERIOD = 26
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70  # Increased from 60 to make it stricter
RSI_OVERSOLD = 30   # Decreased from 40 to make it stricter

# --- TP/SL Settings ---
TP_PERCENTAGE = 0.01  # Take Profit percentage (e.g., 0.01 for 1%)
SL_PERCENTAGE = 0.005 # Stop Loss percentage (e.g., 0.005 for 0.5%)

# --- TIMEZONE SETTINGS ---
IST = pytz.timezone('Asia/Kolkata')  # Indian Standard Timezone
UTC = pytz.utc # UTC Timezone

# --- TRADE MANAGEMENT VARIABLES ---
in_trade = False          # Flag to indicate if a trade is currently active
trade_type = None         # 'BUY' or 'SELL'
entry_price = 0.0
stop_loss = 0.0
take_profit = 0.0
TRADE_STATUS_FILE = "trade_status.txt" # File to store trade status

# --- ANSI COLOR CODES (GLOBAL SCOPE) ---
GREEN = '\033[92m'
RED = '\033[91m'
RESET_COLOR = '\033[0m'
GRAY_BOLD = '\033[97;1m' # White-gray and bold
RESET_BOLD = '\033[0m' # Reset bold (and color) - using RESET_COLOR for simplicity


def fetch_data(symbol, interval):
    """
    Fetches historical data for a given symbol and interval using yfinance.
    """
    df = yf.download(symbol, period="2d", interval=interval) # Fetching 2 days of data
    if df.empty:
        print(f"Failed to fetch data for {symbol} with interval {interval}")
        return None
    return df

def calculate_indicators(df):
    """
    Calculates Moving Averages, MACD, and RSI indicators.
    """
    # --- Moving Averages ---
    df['MA_Fast'] = df['Close'].rolling(window=MOVING_AVERAGE_FAST_PERIOD).mean()
    df['MA_Slow'] = df['Close'].rolling(window=MOVING_AVERAGE_SLOW_PERIOD).mean()

    # --- MACD ---
    EMA_Fast = df['Close'].ewm(span=MACD_FAST_PERIOD, adjust=False).mean()
    EMA_Slow = df['Close'].ewm(span=MACD_SLOW_PERIOD, adjust=False).mean()
    df['MACD'] = EMA_Fast - EMA_Slow
    df['Signal'] = df['MACD'].ewm(span=MACD_SIGNAL_PERIOD, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']

    # --- RSI ---
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=RSI_PERIOD).mean()
    avg_loss = loss.rolling(window=RSI_PERIOD).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df

def save_trade_status():
    """Saves the current trade status to a file."""
    try:
        with open(TRADE_STATUS_FILE, 'w') as f:
            f.write(f"in_trade={in_trade}\n")
            f.write(f"trade_type={trade_type}\n")
            f.write(f"entry_price={entry_price}\n")
            f.write(f"stop_loss={stop_loss}\n")
            f.write(f"take_profit={take_profit}\n")
        print("Trade status saved to file.")
    except Exception as e:
        print(f"Error saving trade status to file: {e}")

def load_trade_status():
    """Loads trade status from file if available."""
    global in_trade, trade_type, entry_price, stop_loss, take_profit
    try:
        if os.path.exists(TRADE_STATUS_FILE):
            with open(TRADE_STATUS_FILE, 'r') as f:
                lines = f.readlines()
                in_trade = lines[0].split('=')[1].strip() == 'True'
                trade_type = lines[1].split('=')[1].strip()
                entry_price = float(lines[2].split('=')[1].strip())
                stop_loss = float(lines[3].split('=')[1].strip())
                take_profit = float(lines[4].split('=')[1].strip())
            print("Trade status loaded from file.")
            return True
        return False # No trade status loaded
    except FileNotFoundError:
        return False # No trade status file found (first run or file deleted)
    except Exception as e:
        print(f"Error loading trade status from file: {e}")
        return False


def clear_trade_status():
    """Clears the trade status file."""
    global in_trade, trade_type, entry_price, stop_loss, take_profit
    try:
        if os.path.exists(TRADE_STATUS_FILE):
            os.remove(TRADE_STATUS_FILE)
            print("Trade status file cleared (trade closed).")
    except Exception as e:
        print(f"Error clearing trade status file: {e}")
    in_trade = False
    trade_type = None
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0


def check_signals(df, symbol):
    """
    Checks for bull and bear signals based on MA, MACD, and RSI indicators with RSI confirmation.
    Modified to ensure Bull and Bear signals are mutually exclusive and to manage trades.
    Added color-coding for signals and active trades in console output.
    Color-coded only "BUY" and "SELL" text in "TRADE ACTIVE" line.
    Styled price, SL, and TP in "TRADE ACTIVE" to white-gray and bold.
    Persists trade status to file.
    """
    global in_trade, trade_type, entry_price, stop_loss, take_profit, GREEN, RED, RESET_COLOR, GRAY_BOLD, RESET_BOLD # Declare global variables and colors

    current_time_utc = datetime.now(UTC)
    current_time_ist = current_time_utc.astimezone(IST)
    current_time_str_utc = current_time_utc.strftime('%Y-%m-%d %H:%M:%S%Z')
    current_time_str_ist = current_time_ist.strftime('%Y-%m-%d %I:%M:%S %p %Z')


    # --- Get Current Price for Live Update ---
    current_price = float(df['Close'].iloc[-1].item())

    # --- MA Conditions ---
    ma_fast_current = df['MA_Fast'].iloc[-1]
    ma_slow_current = df['MA_Slow'].iloc[-1]
    ma_fast_previous = df['MA_Fast'].iloc[-2]
    ma_slow_previous = df['MA_Slow'].iloc[-2]

    ma_condition_bull = ma_fast_current > ma_slow_current or (ma_fast_current >= ma_slow_current and ma_fast_previous < ma_slow_previous)
    ma_condition_bear = ma_fast_current < ma_slow_current or (ma_fast_current <= ma_slow_current and ma_fast_previous > ma_slow_previous)

    ma_values_str = f"Fast MA={ma_fast_current:.2f}, Slow MA={ma_slow_current:.2f}"

    # --- MACD Conditions ---
    macd_current = df['MACD'].iloc[-1]
    signal_current = df['Signal'].iloc[-1]
    macd_hist = df['MACD_Hist'].iloc[-1]

    macd_condition_bull = macd_current > signal_current or macd_hist > 0
    macd_condition_bear = macd_current < signal_current or macd_hist < 0

    macd_values_str = f"MACD={macd_current:.2f}, Signal={signal_current:.2f}, Histogram={macd_hist:.2f}"

    # --- RSI Conditions ---
    rsi_value = df['RSI'].iloc[-1]
    rsi_oversold_strong = rsi_value < RSI_OVERSOLD - 10
    rsi_overbought_strong = rsi_value > RSI_OVERBOUGHT + 10
    rsi_condition_bull_signal = rsi_value <= RSI_OVERBOUGHT
    rsi_condition_bear_signal = rsi_value >= RSI_OVERSOLD

    trend_band_trend = "Neutral"
    ma_trend = "Neutral"

    bull_signal = False
    bear_signal = False

    # --- Check for Active Trade Before Generating New Signals ---
    if not in_trade: # Only check for new signals if not already in a trade
        # --- Bull Signal Check (PRIORITIZED) ---
        print(f"""
    --- Bull Signal Check for {symbol} at {current_time_str_utc} ---
    MA Fast > Slow OR Close to Crossover: {ma_condition_bull}, MA Values: {ma_values_str}
    MACD > Signal OR MACD Hist > 0: {macd_condition_bull}, MACD Values: {macd_values_str}
    RSI Oversold (for STRONG signal) or NOT Overbought (for Regular Signal): {rsi_condition_bull_signal}, RSI Value: df['RSI'].iloc[-1]={rsi_value:.2f}, Oversold Level: RSI_OVERSOLD={RSI_OVERSOLD}, Overbought Level: RSI_OVERBOUGHT={RSI_OVERBOUGHT}
    """)
        print(f"Debug: MA Bull Condition: {ma_condition_bull}")
        print(f"Debug: MACD Bull Condition: {macd_condition_bull}")
        print(f"Debug: RSI NOT Overbought Condition (Bull): {rsi_condition_bull_signal}")

        bull_signal = (ma_condition_bull or macd_condition_bull) and rsi_condition_bull_signal
        print(f"Debug: Final Bull Signal: {bull_signal}")


        if bull_signal: # Check for Bull signal first
            entry_price = float(df['Close'].iloc[-1].item())
            stop_loss = float(entry_price * (1 - SL_PERCENTAGE))
            take_profit = float(entry_price * (1 + TP_PERCENTAGE))

            signal_type = "STRONG Signal with RSI Confirmation" if rsi_oversold_strong else "Regular Signal with RSI Confirmation"

            # --- ENTER BUY TRADE ---
            in_trade = True
            trade_type = 'BUY' # Set trade type
            save_trade_status() # SAVE TRADE STATUS TO FILE
            print(f"""
    {GREEN}🔥🔥🔥 BULL SIGNAL generated for {symbol} at {current_time_str_utc}! 🔥🔥🔥{RESET_COLOR} ({signal_type})
      🔥 NEW SIGNAL: BUY at {entry_price:.2f} 🔥
        Stop Loss (SL): {stop_loss:.2f}
        Take Profit (TP): {take_profit:.2f}""")

        else: # If no Bull signal, then check for Bear signal
            # --- Bear Signal Check ---
            print(f"""
    --- Bear Signal Check for {symbol} at {current_time_str_utc} ---
    MA Fast < Slow OR Close to Crossover: {ma_condition_bear}, MA Values: {ma_values_str}
    MACD < Signal OR MACD Hist < 0: {macd_condition_bear}, MACD Values: {macd_values_str}
    RSI Overbought (for STRONG signal) or NOT Oversold (for Regular Signal): {rsi_condition_bear_signal}, RSI Value: df['RSI'].iloc[-1]={rsi_value:.2f}, Overbought Level: RSI_OVERBOUGHT={RSI_OVERBOUGHT}, Oversold Level: RSI_OVERSOLD={RSI_OVERSOLD}
    """)
            print(f"Debug: MA Bear Condition: {ma_condition_bear}")
            print(f"Debug: MACD Bear Condition: {macd_condition_bear}")
            print(f"Debug: RSI NOT Oversold Condition (Bear): {rsi_condition_bear_signal}")

            bear_signal = (ma_condition_bear or macd_condition_bear) and rsi_condition_bear_signal
            print(f"Debug: Final Bear Signal: {bear_signal}")

            if bear_signal:
                entry_price = float(df['Close'].iloc[-1].item())
                stop_loss = float(entry_price * (1 + SL_PERCENTAGE))
                take_profit = float(entry_price * (1 - TP_PERCENTAGE))

                signal_type = "STRONG Signal with RSI Confirmation" if rsi_overbought_strong else "Regular Signal with RSI Confirmation"

                # --- ENTER SELL TRADE ---
                in_trade = True
                trade_type = 'SELL' # Set trade type
                save_trade_status() # SAVE TRADE STATUS TO FILE
                print(f"""
    {RED}🔥🔥🔥 BEAR SIGNAL generated for {symbol} at {current_time_str_utc}! 🔥🔥🔥{RESET_COLOR} ({signal_type})
      🔥 NEW SIGNAL: SELL at {entry_price:.2f} 🔥
        Stop Loss (SL): {stop_loss:.2f}
        Take Profit (TP): {take_profit:.2f}""")
            else:
                print(f"No Bear Signal for {symbol} at {current_time_str_utc}")
    else:
        print("Trade already active, skipping signal check.") # Indicate that signal check is skipped because of active trade


    # --- Live Trade Management (Check TP/SL if in a trade) ---
    if in_trade:
        if trade_type == 'BUY':
            if current_price <= stop_loss: # Stop Loss hit for BUY trade
                print(f"🛑 STOP LOSS HIT for BUY trade at {current_time_str_utc}! 🛑 Price: {current_price:.2f}, SL: {stop_loss:.2f}")
                clear_trade_status() # CLEAR TRADE STATUS FILE
            elif current_price >= take_profit: # Take Profit hit for BUY trade
                print(f"✅ TAKE PROFIT HIT for BUY trade at {current_time_str_utc}! ✅ Price: {current_price:.2f}, TP: {take_profit:.2f}")
                clear_trade_status() # CLEAR TRADE STATUS FILE
        elif trade_type == 'SELL':
            if current_price >= stop_loss: # Stop Loss hit for SELL trade (price moved up)
                print(f"🛑 STOP LOSS HIT for SELL trade at {current_time_str_utc}! 🛑 Price: {current_price:.2f}, SL: {stop_loss:.2f}")
                clear_trade_status() # CLEAR TRADE STATUS FILE
            elif current_price <= take_profit: # Take Profit hit for SELL trade (price moved down)
                print(f"✅ TAKE PROFIT HIT for SELL trade at {current_time_str_utc}! ✅ Price: {current_price:.2f}, TP: {take_profit:.2f}")
                clear_trade_status() # CLEAR TRADE STATUS FILE


    # --- Live Update Section (Always Print) ---
    print(f"Debug: Type of current_price: {type(current_price)}")
    print(f"Debug: Type of rsi_value: {type(rsi_value)}")
    print(f"Debug: Type of macd_hist: {type(macd_hist)}")
    print(f"""
----- Live Update: {current_time_str_ist} -----

--- Symbol: {symbol} ---
  Price: {current_price:.2f}
  Trend Band Trend: {trend_band_trend}
  MA Trend: {ma_trend}
  RSI: {rsi_value:.2f}
  MACD Histogram: {macd_hist:.3f}""")

    if in_trade: # Display active trade details in Live Update if in a trade
        if trade_type == 'BUY':
            print(f"""
  📈 TRADE ACTIVE: {GREEN}BUY{RESET_COLOR} at {GRAY_BOLD}{entry_price:.2f}{RESET_BOLD} 📉
    Stop Loss (SL): {GRAY_BOLD}{stop_loss:.2f}{RESET_BOLD}
    Take Profit (TP): {GRAY_BOLD}{take_profit:.2f}{RESET_BOLD}""")
        elif trade_type == 'SELL':
            print(f"""
  📈 TRADE ACTIVE: {RED}SELL{RESET_COLOR} at {GRAY_BOLD}{entry_price:.2f}{RESET_BOLD} 📉
    Stop Loss (SL): {GRAY_BOLD}{stop_loss:.2f}{RESET_BOLD}
    Take Profit (TP): {GRAY_BOLD}{take_profit:.2f}{RESET_BOLD}""")

    elif bull_signal: # Print Buy signal details in Live Update if Bull signal is active (and no trade is yet active - trade starts in next iteration)
        print(f"""
  {GREEN}🔥 SIGNAL: BUY at {entry_price:.2f} 🔥{RESET_COLOR}
    Stop Loss (SL): {stop_loss:.2f}
    Take Profit (TP): {take_profit:.2f}""")
    elif bear_signal: # Print Sell signal details in Live Update if Bear signal is active (and no trade is yet active - trade starts in next iteration)
        print(f"""
  {RED}🔥 SIGNAL: SELL at {entry_price:.2f} 🔥{RESET_COLOR}
    Stop Loss (SL): {stop_loss:.2f}
    Take Profit (TP): {take_profit:.2f}""")
    else:
        print("  No Signal")

    print("----------------------")



def live_monitoring_with_signals(symbols, interval):
    """
    Monitors symbols live, calculates indicators, checks for signals, and manages trades.
    Loads trade status from file at start.
    """
    print(f"yfinance version: {yf.__version__}")
    print("\nStarting live monitoring for multiple symbols with advanced signals and trade management (with debugging)...")

    global in_trade, trade_type, entry_price, stop_loss, take_profit, GREEN, RESET_COLOR, RED, GRAY_BOLD, RESET_BOLD # Use global variables for colors too

    load_trade_status() # LOAD TRADE STATUS FROM FILE

    if in_trade:
        print(f"{GREEN}Resuming ACTIVE TRADE: {trade_type} at {entry_price:.2f}, SL={stop_loss:.2f}, TP={take_profit:.2f}{RESET_COLOR}")
    else:
        print("No active trade loaded from file.")


    while True:
        print("\n--- Starting new loop iteration ---")
        print("Fetching data...")
        data = fetch_data(symbols, interval)
        if data is None:
            print("Failed to fetch data. Retrying in 60 seconds...")
            time.sleep(60)
            continue

        print("Data fetched.")

        print("--- Starting calculate_indicators ---")
        df_with_indicators = calculate_indicators(data.copy())

        print("--- Finished calculate_indicators ---")

        print(f"\n--- Signal Checks for {SYMBOL} at {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S%Z')} ---")

        check_signals(df_with_indicators, SYMBOL)

        print("\nIndicators calculated and signals generated/trade managed.")
        print("Plotting data... (plotting is commented out in this script to focus on signals)")
        # --- Plotting (Optional - you can uncomment this section if you want to plot) ---
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(12, 6))
        # plt.plot(df_with_indicators['Close'], label='Close Price')
        # plt.plot(df_with_indicators['MA_Fast'], label=f'MA Fast ({MOVING_AVERAGE_FAST_PERIOD})')
        # plt.plot(df_with_indicators['MA_Slow'], label=f'MA Slow ({MOVING_AVERAGE_SLOW_PERIOD})')
        # plt.legend(loc='upper left')
        # plt.title(f'{symbol} Price with Moving Averages')
        # plt.xlabel('Time')
        # plt.ylabel('Price')
        # plt.show()
        print("Data plotted. (plotting is commented out in this script)")


        current_time_ist_display = datetime.now(IST).strftime('%Y-%m-%d %I:%M:%S %p %Z')
        print(f"\n----- Live Update: {current_time_ist_display} -----\n")


        time.sleep(60) # Check every 60 seconds


if __name__ == "__main__":
    live_monitoring_with_signals(SYMBOL, TIME_INTERVAL)
