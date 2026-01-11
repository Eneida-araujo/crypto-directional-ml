import pandas as pd
import numpy as np


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw OHLCV data."""
    return pd.read_csv(filepath)

def preprocess_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Convert datetime columns and sort data."""
    df = df.copy()

    df["open_time"] = pd.to_datetime(df["open_time"])
    df["close_time"] = pd.to_datetime(df["close_time"])

    df = df.sort_values("open_time").reset_index(drop=True)

    return df

def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove unused columns."""
    return df.drop(columns=["ignore"])

def create_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Create log returns based on close prices."""
    df = df.copy()

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    return df

def create_rolling_volatility(
    df: pd.DataFrame, window: int = 14
) -> pd.DataFrame:
    """Create rolling volatility of log returns."""
    df = df.copy()

    df[f"volatility_{window}"] = (
        df["log_return"]
        .rolling(window=window, min_periods=window)
        .std()
    )

    return df

def create_momentum(
    df: pd.DataFrame, window: int = 10
) -> pd.DataFrame:
    """Create momentum feature."""
    df = df.copy()

    df[f"momentum_{window}"] = df["close"] - df["close"].shift(window)

    return df

def create_volume_features(
    df: pd.DataFrame, window: int = 14
) -> pd.DataFrame:
    """Create rolling volume features."""
    df = df.copy()

    df[f"volume_mean_{window}"] = (
        df["volume"]
        .rolling(window=window, min_periods=window)
        .mean()
    )

    df[f"volume_std_{window}"] = (
        df["volume"]
        .rolling(window=window, min_periods=window)
        .std()
    )

    return df

def create_directional_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary directional target (t+1)."""
    df = df.copy()

    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    return df

def finalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with NaNs after feature and target creation."""
    df = df.copy()

    df = df.dropna().reset_index(drop=True)

    return df

def build_feature_dataset(filepath: str) -> pd.DataFrame:
    """Build final feature dataset ready for modeling."""
    df = load_raw_data(filepath)
    df = preprocess_datetime(df)
    df = drop_unused_columns(df)

    df = create_log_returns(df)
    df = create_rolling_volatility(df, window=14)
    df = create_momentum(df, window=10)
    df = create_volume_features(df, window=14)

    df = create_directional_target(df)
    df = finalize_dataset(df)

    return df
