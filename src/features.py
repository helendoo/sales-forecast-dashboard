import pandas as pd

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.sort_values("Date")

    df = df.rename(columns={"Sale": "y"})

    df["dayofweek"] = df["Date"].dt.dayofweek # 0 = Monday
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df["weekofyear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # --- Lag features ---
    for lag in [1, 7, 14, 30]:
        df[f"y_lag_{lag}"] = df["y"].shift(lag)

    # --- Rolling window features ---
    for window in [7, 14, 30]:
        df[f"y_roll_mean_{window}"] = df["y"].shift(1).rolling(window).mean()
        df[f"y_roll_std_{window}"] = df["y"].shift(1).rolling(window).std()

    # --- Temperature lags ---
    for lag in [1, 7, 14]:
        df[f"temp_lag_{lag}"] = df["Temperature"].shift(lag)

    # --- Temperature rolling values ---
    df["temp_roll_mean_7"] = df["Temperature"].shift(1).rolling(7).mean()
    df["temp_roll_mean_30"] = df["Temperature"].shift(1).rolling(30).mean()

    # Drop rows with NaN created by lags
    df = df.dropna().reset_index(drop=True)

    return df
