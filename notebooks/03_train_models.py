import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor
import joblib


feat = pd.read_csv("data/processed/features.csv", parse_dates=["Date"])
print("Loaded features:")
print(feat.head())

y = feat["y"]
X = feat.drop(columns=["y", "Date"])


######################  TRAIN/VALIDATE/TEST    ##################

n = len(feat)
train_end = int(n * 0.7)
val_end   = int(n * 0.85)

X_train, y_train = X[:train_end], y[:train_end]
X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
X_test,  y_test  = X[val_end:], y[val_end:]

print(f"Train: {len(X_train)} rows")
print(f"Val:   {len(X_val)} rows")
print(f"Test:  {len(X_test)} rows")


######################   BASELINE MODELS    ##################


def mae(a, b):
    return mean_absolute_error(a, b)


naive_pred = y_val.shift(1).fillna(method='bfill')
naive_mae = mae(y_val, naive_pred)
print("\nNaive Baseline MAE:", naive_mae)

moving_pred = y.rolling(7).mean().shift(1).iloc[train_end:val_end]
moving_pred = moving_pred.fillna(method="bfill")
moving_mae = mae(y_val, moving_pred)
print("7-day Moving Average MAE:", moving_mae)


######################   TRAIN LIGHTGBM MODEL    ##################

print("\nTraining LightGBM...")

model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="l1",
)

# Validation predictions
val_pred = model.predict(X_val)
val_mae = mae(y_val, val_pred)
print("\nLightGBM MAE (val):", val_mae)


######################   TEST EVALUATION    ##################

test_pred = model.predict(X_test)
test_mae = mae(y_test, test_pred)
print("LightGBM MAE (test):", test_mae)

# Save test predictions for inspection
test_results = pd.DataFrame({
    "Date": feat["Date"].iloc[val_end:].values,
    "Actual": y_test.values,
    "Predicted": test_pred
})
test_results.to_csv("data/processed/test_predictions.csv", index=False)

print("\nSaved test predictions → data/processed/test_predictions.csv")


######################   SAVE MODEL    ##################

joblib.dump(model, "models/lightgbm_model.pkl")
print("Saved model → models/lightgbm_model.pkl")
