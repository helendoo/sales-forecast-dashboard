import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/test_predictions.csv", parse_dates=["Date"])

plt.figure(figsize=(12,4))
plt.plot(df["Date"], df["Actual"], label="Actual")
plt.plot(df["Date"], df["Predicted"], label="Predicted")
plt.title("Actual vs Predicted Sales (Test Set)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.show()
