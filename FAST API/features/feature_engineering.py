"""
Feature engineering for price prediction models
"""
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build engineered features for model prediction.
    
    Args:
        df (pd.DataFrame): Input DataFrame with columns: date, price, and other product features
        
    Returns:
        pd.DataFrame: DataFrame with engineered features including:
                     - Time features: day, month, dayofweek, hour
                     - Lag features: price_lag_1, price_lag_2
                     - Price change features: price_diff_1, price_pct_change_1
                     - Rolling statistics: rolling_mean_3, rolling_min_3, 
                       rolling_max_3, rolling_std_3
                     - Relative features: price_vs_roll_mean, price_vs_roll_min
    
    Note:
        The function creates a copy of the input DataFrame to avoid side effects.
    """
    df = df.copy()
    
    # Convert date to datetime
    df["date"] = pd.to_datetime(df["date"])

    # Time-based features
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    df["hour"] = df["date"].dt.hour

    # Lag features (previous prices)
    df["price_lag_1"] = df["price"].shift(1)
    df["price_lag_2"] = df["price"].shift(2)

    # Price change features
    df["price_diff_1"] = df["price"] - df["price_lag_1"]
    df["price_pct_change_1"] = df["price"].pct_change()

    # Rolling statistics (3-period windows)
    df["rolling_mean_3"] = df["price"].rolling(3).mean()
    df["rolling_min_3"] = df["price"].rolling(3).min()
    df["rolling_max_3"] = df["price"].rolling(3).max()
    df["rolling_std_3"] = df["price"].rolling(3).std()

    # Relative features (price deviation from rolling stats)
    df["price_vs_roll_mean"] = df["price"] - df["rolling_mean_3"]
    df["price_vs_roll_min"] = df["price"] - df["rolling_min_3"]

    return df
