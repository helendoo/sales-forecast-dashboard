# 05_forecast_future.py
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt


##################  LOAD DATA + MODEL   ##################

print("Loading data and model...")
df = pd.read_csv("data/processed/daily_sales.csv")
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)  

feat = pd.read_csv("data/processed/features.csv", parse_dates=["Date"])
train_cols = [c for c in feat.columns if c not in ["y", "Date"]]

model = joblib.load("models/lightgbm_model.pkl")


##################  FUTURE DATE RANGE   ##################

last_date = df["Date"].max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                             periods=180, freq="D")
future_df = pd.DataFrame({"Date": future_dates})

print(f"Forecasting from {future_dates[0].date()} to {future_dates[-1].date()}")


##################  TEMPERATURE ASSUMPTION   ##################

rolling_temp = df["Temperature"].tail(30).mean()
future_df["Temperature"] = rolling_temp
print(f"Future temperature assumption: {rolling_temp:.2f}")


##################  BASE TABLE + STATIC TIME FEATURES   ##################

all_data = pd.concat([df[["Date", "Sale", "Temperature"]], future_df], ignore_index=True)

def add_time_features(_df: pd.DataFrame) -> pd.DataFrame:
    _df = _df.copy()
    _df["dayofweek"]  = _df["Date"].dt.dayofweek
    _df["month"]      = _df["Date"].dt.month
    _df["day"]        = _df["Date"].dt.day
    _df["weekofyear"] = _df["Date"].dt.isocalendar().week.astype(int)
    _df["is_weekend"] = (_df["dayofweek"] >= 5).astype(int)
    return _df

all_data = add_time_features(all_data)


all_data["y"] = list(df["Sale"]) + [np.nan] * len(future_df)


for col in train_cols:
    if col not in all_data.columns:
        all_data[col] = np.nan


##################  ITERATIVE FORECAST   ##################

print("Generating 6-month forecast...")

n_hist = len(df) 

for i in range(len(future_df)):
    idx = n_hist + i


    for lag in [1, 7, 14, 30]:
        all_data.loc[idx, f"y_lag_{lag}"] = all_data.loc[idx - lag, "y"]

    for window in [7, 14, 30]:
        all_data.loc[idx, f"y_roll_mean_{window}"] = (
            all_data["y"].shift(1).rolling(window).mean().iloc[idx]
        )
        all_data.loc[idx, f"y_roll_std_{window}"] = (
            all_data["y"].shift(1).rolling(window).std().iloc[idx]
        )

    for lag in [1, 7, 14]:
        all_data.loc[idx, f"temp_lag_{lag}"] = all_data.loc[idx - lag, "Temperature"]

    all_data.loc[idx, "temp_roll_mean_7"] = (
        all_data["Temperature"].shift(1).rolling(7).mean().iloc[idx]
    )
    all_data.loc[idx, "temp_roll_mean_30"] = (
        all_data["Temperature"].shift(1).rolling(30).mean().iloc[idx]
    )

    X_pred = all_data.loc[idx, train_cols].values.reshape(1, -1)
    y_pred = model.predict(X_pred)[0]
    all_data.loc[idx, "y"] = y_pred 


##################  SAVE RESULTS   ##################

future_predictions = all_data.iloc[n_hist:][["Date", "y"]].copy()
future_predictions.columns = ["Date", "Predicted"]

out_path = Path("data/processed/future_forecast.csv")
future_predictions.to_csv(out_path, index=False)
print(f"\nSaved 6-month forecast → {out_path}")




##################  PLOT RESULTS   ##################

plt.figure(figsize=(14, 5))
plt.plot(df["Date"], df["Sale"], label="Historical Sales")
plt.plot(future_predictions["Date"], future_predictions["Predicted"], label="Forecast")
plt.title("6-Month Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.show()
