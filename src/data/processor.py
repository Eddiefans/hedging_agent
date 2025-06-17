import os
import yfinance as yf
import pandas as pd
import numpy as np
import ta

def download_raw_data(ticker, start_date, end_date, output_path=None, verbose=True):
    """
    Download raw historical price data from Yahoo Finance.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        output_path: Path to save the raw data CSV file
        verbose: Whether to print progress messages 
        
    Returns:
        pandas.DataFrame: Raw historical price data 
    """
    if verbose:
        print("Downloading raw data for {} from {} to {}...".format(ticker, start_date, end_date))
    
    # Download data from Yahoo Finance
    raw = yf.download(ticker, start=start_date, end=end_date, threads=False)
    
    # Save if there is an output path
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        raw.to_csv(output_path)
        if verbose:
            print(f"Raw data saved to {output_path}")
    
    return raw

def create_hedging_features_dataset(
    ticker,
    lags=10,
    start_date="2015-01-01",
    end_date="2024-12-31",
    output_path=None,
    raw_data_path=None,
    sma_windows=[10, 20, 50],
    ema_windows=[12, 26, 50],
    rsi_window=14,
    macd_fast=12,
    macd_slow=26,
    macd_signal=9,
    bb_window=20,
    bb_std=2,
    atr_window=14,
    momentum_windows=[5, 10, 20],
    volume_windows=[10, 20],
    volatility_windows=[10, 20, 30],
    verbose=True,
):
    """
    Create a comprehensive dataset with technical indicators for portfolio hedging.
    
    Args:
        ticker: Stock ticker symbol (default NVDA)
        lags: Number of lagged returns to include
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        output_path: Path to save the processed dataset
        raw_data_path: Path to load raw data from (if None, download fresh)
        sma_windows: List of Simple Moving Average windows
        ema_windows: List of Exponential Moving Average windows
        rsi_window: Relative Strength Index window
        macd_fast: MACD fast period
        macd_slow: MACD slow period
        macd_signal: MACD signal period
        bb_window: Bollinger Bands window
        bb_std: Bollinger Bands standard deviation
        atr_window: Average True Range window
        momentum_windows: List of momentum indicator windows
        volume_windows: List of volume indicator windows
        volatility_windows: List of volatility windows for features
        verbose: Whether to print progress messages
    
    Returns:
        pandas.DataFrame: Processed dataset with technical indicators
    """
    
    # Either load raw data from file or download it
    if raw_data_path and os.path.exists(raw_data_path):
        if verbose:
            print("Loading raw data from {}".format(raw_data_path))
        raw_price_data = pd.read_csv(raw_data_path, index_col=0, parse_dates=True)
        
        # Ensure the OHLCV columns are numeric
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in raw_price_data.columns:
                raw_price_data[col] = pd.to_numeric(raw_price_data[col], errors='coerce')
        
        # Download VIX and market indices separately for market context
        if verbose:
            print("Downloading market context data (VIX, SPY)...")
        market_data = yf.download(["^VIX", "SPY"], start=start_date, end=end_date, threads=False)
        vix = market_data["Close"]["^VIX"] if "^VIX" in market_data["Close"].columns else None
        spy = market_data["Close"]["SPY"] if "SPY" in market_data["Close"].columns else None
    else:
        # Download ticker, VIX, and SPY in one call for context
        if verbose:
            print("Downloading data for {}, VIX, and SPY...".format(ticker))
        symbols = [ticker, "^VIX", "SPY"]
        raw = yf.download(symbols, start=start_date, end=end_date, threads=False)
        
        # Extract main stock data
        if len(symbols) > 1:
            raw_price_data = raw.xs(ticker, axis=1, level=1)[["Open", "High", "Low", "Close", "Volume"]]
            vix = raw["Close"]["^VIX"] if "^VIX" in raw["Close"].columns else None
            spy = raw["Close"]["SPY"] if "SPY" in raw["Close"].columns else None
        else:
            raw_price_data = raw[["Open", "High", "Low", "Close", "Volume"]]
            vix = None
            spy = None

    # Create working dataframe
    df = pd.DataFrame(index=raw_price_data.index)
    df.index.name = "Date"
    df["Close"] = raw_price_data["Close"]
    df["Open"] = raw_price_data["Open"]
    df["High"] = raw_price_data["High"]
    df["Low"] = raw_price_data["Low"]
    df["Volume"] = raw_price_data["Volume"]
    
    # Ensure all price columns are numeric
    for col in ["Close", "Open", "High", "Low", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()

    if verbose:
        print("Computing technical indicators for hedging...")
    
    # Price-based features
    df["returns"] = df["Close"].pct_change()
    df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))
    
    # Lagged returns for momentum/mean reversion signals
    for lag in range(1, lags + 1):
        df[f"returns_lag_{lag}"] = df["returns"].shift(lag)
        df[f"log_returns_lag_{lag}"] = df["log_returns"].shift(lag)

    # Multiple timeframe moving averages
    for window in sma_windows:
        df[f"SMA_{window}"] = ta.trend.sma_indicator(df["Close"], window=window)
        df[f"price_sma_{window}_ratio"] = df["Close"] / df[f"SMA_{window}"]
        
    for window in ema_windows:
        df[f"EMA_{window}"] = ta.trend.ema_indicator(df["Close"], window=window)
        df[f"price_ema_{window}_ratio"] = df["Close"] / df[f"EMA_{window}"]

    # RSI for overbought/oversold conditions
    df["RSI"] = ta.momentum.rsi(df["Close"], window=rsi_window)
    df["RSI_normalized"] = (df["RSI"] - 50) / 50  # Normalize around 0
    
    # MACD for trend detection
    macd = ta.trend.MACD(
        df["Close"],
        window_slow=macd_slow,
        window_fast=macd_fast,
        window_sign=macd_signal,
    )
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_histogram"] = macd.macd_diff()
    df["MACD_signal_strength"] = df["MACD"] - df["MACD_signal"]
    
    # Bollinger Bands for volatility and mean reversion
    bb = ta.volatility.BollingerBands(
        df["Close"], window=bb_window, window_dev=bb_std
    )
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()
    df["BB_middle"] = bb.bollinger_mavg()
    df["BB_position"] = (df["Close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_middle"]

    # ATR for volatility
    atr = ta.volatility.AverageTrueRange(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=atr_window,
    )
    df["ATR"] = atr.average_true_range()
    df["ATR_normalized"] = df["ATR"] / df["Close"]  # Normalize by price
    
    # Volume indicators
    df["OBV"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])
    df["VWAP"] = ta.volume.volume_weighted_average_price(
        df["High"], df["Low"], df["Close"], df["Volume"]
    )
    df["price_vwap_ratio"] = df["Close"] / df["VWAP"]
    
    # Volume moving averages
    for window in volume_windows:
        df[f"volume_sma_{window}"] = df["Volume"].rolling(window=window).mean()
        df[f"volume_ratio_{window}"] = df["Volume"] / df[f"volume_sma_{window}"]

    # Momentum indicators
    for window in momentum_windows:
        df[f"momentum_{window}"] = df["Close"] / df["Close"].shift(window) - 1
        df[f"roc_{window}"] = ta.momentum.roc(df["Close"], window=window)

    # Volatility features (important for hedging)
    for window in volatility_windows:
        df[f"volatility_{window}"] = df["returns"].rolling(window=window).std()
        df[f"volatility_{window}_normalized"] = df[f"volatility_{window}"] * np.sqrt(252)  # Annualized
    
    # High-Low spreads and ranges
    df["high_low_spread"] = (df["High"] - df["Low"]) / df["Close"]
    df["open_close_spread"] = (df["Close"] - df["Open"]) / df["Open"]
    
    # Market context features
    if vix is not None:
        df["VIX"] = vix
        df["VIX_change"] = vix.pct_change()
        # VIX relative to its moving average
        df["VIX_sma_20"] = vix.rolling(window=20).mean()
        df["VIX_relative"] = vix / df["VIX_sma_20"]
    
    if spy is not None:
        df["SPY"] = spy
        df["SPY_returns"] = spy.pct_change()
        # Beta-like relationship
        rolling_window = 30
        df["beta_spy"] = df["returns"].rolling(window=rolling_window).cov(
            df["SPY_returns"]
        ) / df["SPY_returns"].rolling(window=rolling_window).var()
        
        # Relative performance vs market
        df["relative_performance"] = df["returns"] - df["SPY_returns"]
    
    # Cross-indicator features
    df["rsi_bb_signal"] = df["RSI_normalized"] * df["BB_position"]
    df["macd_momentum_signal"] = df["MACD_histogram"] * df["momentum_10"]
    
    # Time-based features (useful for regime detection)
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["day_of_week"] = df.index.dayofweek
    df["week_of_year"] = df.index.isocalendar().week
    
    # Seasonal dummies
    for month in range(1, 13):
        df[f"month_{month}"] = (df["month"] == month).astype(int)
    
    for dow in range(5):  # Monday=0 to Friday=4
        df[f"dow_{dow}"] = (df["day_of_week"] == dow).astype(int)
    
    # Drop intermediate columns we don't need as features
    columns_to_drop = [
        "Open", "High", "Low", "Volume",  # Keep Close for environment
        "month", "quarter", "day_of_week", "week_of_year"  # Replaced by dummies
    ]
    
    # Also drop the individual SMA/EMA values, keep only ratios
    for window in sma_windows:
        columns_to_drop.append(f"SMA_{window}")
    for window in ema_windows:
        columns_to_drop.append(f"EMA_{window}")
    
    # Drop other intermediate calculations
    columns_to_drop.extend([
        "BB_upper", "BB_lower", "BB_middle", "VWAP", "OBV",
        "VIX_sma_20", "SPY"
    ])
    
    # Clean up volume SMAs
    for window in volume_windows:
        columns_to_drop.append(f"volume_sma_{window}")
    
    # Remove columns that exist
    existing_drops = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=existing_drops)
    
    # Drop any remaining NaNs
    df.dropna(inplace=True)
    
    # Reset index to make Date a column
    df.reset_index(inplace=True)
    
    if verbose:
        print(f"Created dataset with {len(df)} rows and {len(df.columns)} columns")
        print(f"Features (excluding Date and Close): {len(df.columns) - 2}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Save processed data if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        if verbose:
            print("Processed features saved to {}".format(output_path))
    
    return df

if __name__ == "__main__":
    # Usage for NVIDIA (NVDA) stock data
    ticker = 'NVDA'
    start_date = '2015-01-01'  # More recent data for better ML features
    end_date = '2024-12-31'
    
    # Create directory structure if it doesn't exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Define paths
    raw_path = "data/raw/{}_raw_{}_{}.csv".format(ticker, start_date, end_date)
    processed_path = "data/processed/{}_hedging_features.csv".format(ticker)
    
    # Download raw data
    raw_data = download_raw_data(ticker, start_date, end_date, raw_path)
    
    # Process data and create hedging features
    features_df = create_hedging_features_dataset(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        output_path=processed_path,
        raw_data_path=raw_path
    )
    
    print("\n=== Dataset Summary ===")
    print(f"Shape: {features_df.shape}")
    print(f"Columns: {list(features_df.columns)}")
    print(f"Date range: {features_df['Date'].min()} to {features_df['Date'].max()}")
