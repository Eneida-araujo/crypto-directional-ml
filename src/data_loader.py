from datetime import datetime
from typing import Optional

import pandas as pd
from binance.client import Client


def create_binance_client() -> Client:
    """Create a Binance client without authentication."""
    return Client()


def load_binance_ohlcv(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Download historical OHLCV data from Binance."""
    client = create_binance_client()

    end_date = end_date or datetime.today().strftime("%Y-%m-%d")

    klines = client.get_historical_klines(
        symbol=symbol,
        interval=interval,
        start_str=start_date,
        end_str=end_date,
    )

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]

    return pd.DataFrame(klines, columns=columns)


def preprocess_raw_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Convert timestamps and numeric columns to proper formats."""
    df = df.copy()

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]

    df[numeric_cols] = df[numeric_cols].astype(float)

    return df


def validate_ohlcv_integrity(df: pd.DataFrame) -> None:
    """Validate temporal continuity and OHLC consistency."""
    if df.isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    if not df["open_time"].is_monotonic_increasing:
        raise ValueError("Timestamps are not in chronological order.")

    invalid_ohlc = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    )

    if invalid_ohlc.any():
        raise ValueError("Invalid OHLC values detected.")


def save_raw_data(df: pd.DataFrame, filepath: str) -> None:
    """Save raw OHLCV data to disk."""
    df.to_csv(filepath, index=False)


def main() -> None:
    SYMBOL = "BTCUSDT"
    INTERVAL = Client.KLINE_INTERVAL_4HOUR
    START_DATE = "2019-01-01"

    df = load_binance_ohlcv(
        symbol=SYMBOL,
        interval=INTERVAL,
        start_date=START_DATE,
    )

    df = preprocess_raw_ohlcv(df)
    validate_ohlcv_integrity(df)

    save_raw_data(
        df,
        filepath="data/raw/btcusdt_4h_2019_present.csv",
    )

    print("Arquivo salvo com sucesso.")


if __name__ == "__main__":
    main()
