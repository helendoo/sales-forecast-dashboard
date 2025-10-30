import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.features import make_features
import pandas as pd



# Load cleaned dataset
df = pd.read_csv("data/processed/daily_sales.csv", parse_dates=["Date"])

print("Loaded data:")
print(df.head())

# Generate the feature table
feat = make_features(df)

print("\nFeature DataFrame:")
print(feat.head())

###################   PLOT TO SEE LAGS   ###################

import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))
plt.plot(feat["Date"], feat["y"], label="Actual Sales")
plt.plot(feat["Date"], feat["y_lag_7"], label="Sales 7 days ago")
plt.legend()
plt.title("Check: Actual Sales vs Sales 7 Days Ago")
plt.xlabel("Date")
plt.ylabel("Sales Value")
plt.show()

# Save processed feature dataset

feat.to_csv("data/processed/features.csv", index=False)
print("\nSaved feature dataset to data/processed/features.csv")
