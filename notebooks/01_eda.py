import pandas as pd
import matplotlib.pyplot as plt

################    HANDLE MISSING VALUES AND DUPLICATES    ################

df = pd.read_csv("data/raw/sales.csv", parse_dates=["Date"])

df['Temperature'].fillna(method='ffill', inplace=True)
df['Sale'].interpolate(inplace=True)

print("HEAD:")
print(df.head())

print("\nINFO:")
print(df.info())

print("\nMissing values:")
print(df.isna().sum())

print("\nSummary statistics:")
print(df.describe())

print("\nDuplicates: ")
print(df.duplicated().sum())


# Save cleaned copy (even if identical)
df.to_csv("data/processed/daily_sales.csv", index=False)




'''
# Basic plot
plt.figure(figsize=(12,4))
plt.plot(df["Date"], df["Sale"])
plt.title("Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Sale")
plt.show()

# Temperature over time
plt.figure(figsize=(12,4))
plt.plot(df["Date"], df["Temperature"], color="orange")
plt.title("Temperature Over Time")
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.show()

# Sales vs temperature scatter
plt.figure(figsize=(6,4))
plt.scatter(df["Temperature"], df["Sale"], alpha=0.4)
plt.title("Sales vs Temperature")
plt.xlabel("Temperature")
plt.ylabel("Sales")
plt.show()

'''

