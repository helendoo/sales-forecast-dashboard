import pandas as pd
import numpy as np
import streamlit as st
import joblib
from lightgbm import LGBMRegressor
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta


##################  HELPERS  ##################

@st.cache_data
def load_data():
    hist = pd.read_csv("data/processed/daily_sales.csv")
    feats = pd.read_csv("data/processed/features.csv")

   
    for df in (hist, feats):
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    return hist.sort_values("Date"), feats.sort_values("Date")

@st.cache_resource
def load_model():
    return joblib.load("models/lightgbm_model.pkl")

def monthly(df):
    m = df.resample("MS", on="Date").sum(numeric_only=True).reset_index()
    return m

def compute_trend_text(hist_m, fc_m):
    # Simple slope: compare last 3 months actual avg vs first 3 months of forecast avg
    if len(hist_m) < 3 or len(fc_m) < 3:
        return "↔ Stable", "gray"
    past = hist_m.tail(3)["Sale"].mean()
    future = fc_m.head(3)["Predicted"].mean()
    change = (future - past) / max(1e-6, past)
    if change > 0.03:
        return "⬆ Slight increase expected", "green"
    if change < -0.03:
        return "⬇ Slight decline expected", "red"
    return "↔ Stable", "gray"

def readable_importances(model, feature_cols):
    # Group raw feature names -> human concepts
    mapping = {
        "Recent sales (last 1–30 days)": [c for c in feature_cols if c.startswith("y_lag_") or c.startswith("y_roll_")],
        "Temperature (recent level & change)": [c for c in feature_cols if c.startswith("temp_lag_") or c.startswith("temp_roll_") or c=="Temperature"],
        "Calendar patterns (month, weekday, weekend)": [c for c in feature_cols if c in {"month","dayofweek","is_weekend","weekofyear","day"}],
    }
    imp = pd.Series(model.feature_importances_, index=feature_cols)
    buckets = []
    for label, cols in mapping.items():
        score = float(imp.get(cols, pd.Series()).sum()) if len(cols)>0 else 0.0
        buckets.append((label, score))
    # Normalize to % for display
    total = sum(s for _, s in buckets) or 1.0
    buckets = [(lbl, round(100*s/total, 1)) for lbl, s in buckets]
    buckets = sorted(buckets, key=lambda x: x[1], reverse=True)
    return buckets

##################  PAGE  ##################

st.set_page_config(page_title="Sales Forecast Dashboard", layout="wide")

hist, feats = load_data()
#hist["Date"] = pd.to_datetime(hist["Date"], dayfirst=True, errors="coerce")
#feats["Date"] = pd.to_datetime(feats["Date"], dayfirst=True, errors="coerce")
model = load_model()

mae_text = "—"
try:
    test_pred = pd.read_csv("data/processed/test_predictions.csv")
    test_pred["Date"] = pd.to_datetime(test_pred["Date"])
    merged = test_pred.merge(hist[["Date","Sale"]].rename(columns={"Sale":"ActualFull"}), on="Date", how="left")
    mae_val = (merged["ActualFull"] - merged["Predicted"]).abs().mean()
    mae_text = f"{mae_val:,.2f}"
    resid_std = (merged["ActualFull"] - merged["Predicted"]).std()
except Exception:
    resid_std = None

##################  HEADER: CONTEXT ##################

st.title("Sales Forecast Dashboard")
st.caption("Forecasting future sales based on historical data and temperature trends.")

with st.expander("How to read this dashboard"):
    st.markdown("""
This dashboard predicts future sales from your past data and temperature.
- **Historical vs Forecast**: Blue shows actual sales to date; orange (dashed) is the forecast. The red line marks where the forecast begins.
- **Key Insights**: Quick facts about horizon, accuracy, trend, and period covered.
- **Drivers**: The main signals the model relied on, translated into plain language.
- **Reliability**: What to trust (more near-term), and current limitations.
""")


left, right = st.columns([1,1])
with left:
    horizon_months = st.slider("Forecast horizon", 1, 6, 6)
with right:
    temp_mode = st.selectbox("Temperature used for forecast",
                             ["Use recent 30-day average", "Repeat same month last year (if available)", "Manual (constant)"])
    manual_temp = None
    if temp_mode == "Manual (constant)":
        manual_temp = float(st.number_input("Manual temperature (°C)", value=float(hist["Temperature"].tail(30).mean())))


##################  BUILD FUTURE TEMPRATURE ##################
last_date = hist["Date"].max()
future_days = int(horizon_months * 30)
future_index = pd.date_range(last_date + pd.Timedelta(days=1), periods=future_days, freq="D")

if temp_mode == "Use recent 30-day average":
    future_temp = pd.Series(hist["Temperature"].tail(30).mean(), index=future_index)
elif temp_mode == "Repeat same month last year (if available)" and (hist["Date"].max() - hist["Date"].min()).days >= 365:
 
    last_year = hist[hist["Date"] >= (hist["Date"].max() - pd.DateOffset(years=1))]
    mtemp = last_year.groupby(last_year["Date"].dt.month)["Temperature"].mean()
    future_temp = pd.Series([mtemp.get(d.month, hist["Temperature"].tail(30).mean()) for d in future_index], index=future_index)
else:
    val = manual_temp if manual_temp is not None else hist["Temperature"].tail(30).mean()
    future_temp = pd.Series(val, index=future_index)


#############  Iterative forecast using trained feature set   ###############

drop_cols = {"y","Date","Sale"}
feature_cols = [c for c in feats.columns if c not in drop_cols]

df_hist = hist.copy().sort_values("Date").reset_index(drop=True)
df_hist["y"] = df_hist["Sale"]

future_df = pd.DataFrame({"Date": future_index, "Temperature": future_temp.values})
future_df["y"] = np.nan
all_df = pd.concat([df_hist[["Date","Temperature","y"]], future_df], ignore_index=True)

all_df["dayofweek"] = all_df["Date"].dt.dayofweek
all_df["month"]     = all_df["Date"].dt.month
all_df["day"]       = all_df["Date"].dt.day
all_df["weekofyear"]= all_df["Date"].dt.isocalendar().week.astype(int)
all_df["is_weekend"]= (all_df["dayofweek"]>=5).astype(int)

for col in feature_cols:
    if col not in all_df.columns:
        all_df[col] = np.nan

for i in range(len(future_df)):
    idx = len(df_hist) + i
    # lags from prior y
    for lag in [1,7,14,30]:
        col = f"y_lag_{lag}"
        if col in all_df.columns:
            all_df.loc[idx, col] = all_df.loc[idx - lag, "y"] if idx - lag >= 0 else np.nan
    # rolling stats
    for w in [7,14,30]:
        colm, cols = f"y_roll_mean_{w}", f"y_roll_std_{w}"
        if colm in all_df.columns:
            all_df.loc[idx, colm] = all_df["y"].shift(1).rolling(w).mean().iloc[idx]
        if cols in all_df.columns:
            all_df.loc[idx, cols] = all_df["y"].shift(1).rolling(w).std().iloc[idx]
    # temp lags
    for lag in [1,7,14]:
        col = f"temp_lag_{lag}"
        if col in all_df.columns:
            all_df.loc[idx, col] = all_df.loc[idx - lag, "Temperature"] if idx - lag >= 0 else np.nan
    if "temp_roll_mean_7" in all_df.columns:
        all_df.loc[idx, "temp_roll_mean_7"]  = all_df["Temperature"].shift(1).rolling(7).mean().iloc[idx]
    if "temp_roll_mean_30" in all_df.columns:
        all_df.loc[idx, "temp_roll_mean_30"] = all_df["Temperature"].shift(1).rolling(30).mean().iloc[idx]

    X_pred = all_df.loc[idx, feature_cols].values.reshape(1, -1)
    all_df.loc[idx, "y"] = float(model.predict(X_pred)[0])

future_pred = all_df.tail(len(future_df))[["Date","y"]].rename(columns={"y":"Predicted"})

# Aggregate to monthly for clean storytelling
hist_m = monthly(hist[["Date","Sale"]])
fc_m   = monthly(future_pred.rename(columns={"Predicted":"Sale"}).rename(columns={"Sale":"Predicted"}))



####################  KEY INSIGHTS (metric cards)   ###################

trend_text, trend_color = compute_trend_text(hist_m, fc_m)


col1, col2, col3, col4 = st.columns(4)
col1.metric("Forecast Horizon", f"{horizon_months} months")
col2.metric("Model Accuracy (MAE)", mae_text, help="Lower is better on the held-out test window")

trend_html = f"""
<div style='font-size:14px; color:{"green" if "increase" in trend_text else "red" if "decline" in trend_text else "gray"};'>
{trend_text}
</div>
"""
period_html = f"""
<div style='font-size:14px; color:gray;'>
{hist['Date'].min().date()} → {hist['Date'].max().date()}
</div>
"""

col3.markdown("**Trend Direction**")
col3.markdown(trend_html, unsafe_allow_html=True)
col4.markdown("**Data Period**")
col4.markdown(period_html, unsafe_allow_html=True)
st.subheader("Sales over time (blue = actuals, orange = forecast)")
split_date = hist["Date"].max()

fig = go.Figure()
fig.add_trace(go.Scatter(x=hist_m["Date"], y=hist_m["Sale"], mode="lines",
                         name="Historical (monthly)", line=dict(color="#1f77b4", width=2)))

if resid_std is None:
    resid_std = hist_m["Sale"].tail(12).std() if len(hist_m)>=12 else hist_m["Sale"].std()

fc_line = fc_m.rename(columns={"Predicted":"y"}).copy()
fc_line["upper"] = fc_line["y"] + 1.25*resid_std
fc_line["lower"] = np.maximum(0, fc_line["y"] - 1.25*resid_std)

fig.add_trace(go.Scatter(x=fc_line["Date"], y=fc_line["upper"],
                         line=dict(width=0), hoverinfo="skip", name="Confidence (+)", showlegend=False))
fig.add_trace(go.Scatter(x=fc_line["Date"], y=fc_line["lower"],
                         fill="tonexty", line=dict(width=0), hoverinfo="skip",
                         name="Forecast confidence", fillcolor="rgba(255,165,0,0.18)"))

fig.add_trace(go.Scatter(x=fc_m["Date"], y=fc_m["Predicted"], mode="lines",
                         name="Forecast (monthly)", line=dict(color="orange", width=2, dash="dash")))

# vertical red line at forecast start
fig.add_vline(x=split_date, line_width=2, line_dash="dot", line_color="red")
fig.add_annotation(x=split_date, y=max(hist_m["Sale"].max(), (fc_m["Predicted"].max() if len(fc_m) else 0)),
                   text="Forecast starts here", showarrow=False, yshift=20, font=dict(color="red"))

fig.update_layout(margin=dict(l=10,r=10,t=10,b=10),
                  xaxis_title="Date", yaxis_title="Sales",
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))

st.plotly_chart(fig, use_container_width=True)
st.caption("The model predicts near-term changes mainly from recent sales patterns and temperature. Reliability decreases further into the future.")





###############    WEEKDAY ACTIVITY (relative index)   ###################

st.subheader("Weekday Activity Index (Mon–Fri = 100)")

period = st.selectbox("Period", ["All data", "Last 8 weeks", "Last 12 months"], index=1)
if period == "Last 8 weeks":
    use_hist = hist[hist["Date"] >= hist["Date"].max() - pd.Timedelta(weeks=8)]
elif period == "Last 12 months":
    use_hist = hist[hist["Date"] >= hist["Date"].max() - pd.DateOffset(years=1)]
else:
    use_hist = hist

weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
tmp = use_hist.assign(
    WeekdayName = use_hist["Date"].dt.day_name(),
    WeekdayNum  = use_hist["Date"].dt.dayofweek
)
wk = (tmp.groupby(["WeekdayNum","WeekdayName"])
          .agg(mean=("Sale","mean"),
               median=("Sale","median"),
               std=("Sale","std"),
               n=("Sale","size"))
          .reset_index())


baseline = wk[wk["WeekdayNum"]<=4]["mean"].mean()
wk["index"] = 100 * wk["mean"] / baseline
wk["sem_index"] = (wk["std"] / wk["n"].pow(0.5)) / baseline * 100  


wk["WeekdayName"] = pd.Categorical(wk["WeekdayName"], categories=weekday_order, ordered=True)
wk = wk.sort_values("WeekdayName")

fig_weekidx = px.bar(
    wk, x="WeekdayName", y="index",
    text=(wk["index"]-100).round(1).astype(str) + "%",
)
fig_weekidx.update_traces(
    textposition="outside",
    error_y=dict(array=wk["sem_index"].fillna(0), visible=True)
)
fig_weekidx.add_hline(y=100, line_dash="dot")  # baseline
fig_weekidx.update_layout(
    margin=dict(l=10,r=10,t=10,b=10),
    xaxis_title="",
    yaxis_title="Weekday Activity Index (Mon–Fri = 100)"
)

st.plotly_chart(fig_weekidx, use_container_width=True)
st.caption("Bars above 100 are busier than the Mon–Fri average. Numbers above bars show % over/under the baseline.")


rank = wk[["WeekdayName","index"]].copy()
rank["% vs baseline"] = (rank["index"] - 100).round(1).astype(str) + "%"
rank = rank.sort_values("index", ascending=False)[["WeekdayName","% vs baseline"]]
st.dataframe(rank.rename(columns={"WeekdayName":"Day"}), use_container_width=True)



##################  DRIVERS OF FORECAST (plain English)    ####################

st.subheader("What’s driving the forecast?")
drivers = readable_importances(model, feature_cols)
drv_df = pd.DataFrame(drivers, columns=["Signal", "Importance (%)"])
drv_fig = px.bar(drv_df, x="Importance (%)", y="Signal", orientation="h",
                 text="Importance (%)", range_x=[0,100])
drv_fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="", xaxis_title="")
drv_fig.update_traces(textposition="outside")
st.plotly_chart(drv_fig, use_container_width=True)
st.caption("Higher bars mean the model relied more on that signal when making predictions.")


###################### FORECAST TABLE (preview)  ##################

st.subheader("Next 10 forecasted periods (monthly view)")
preview = fc_m.copy()
preview["Forecasted Sales"] = preview["Predicted"].round(1)
preview = preview[["Date","Forecasted Sales"]].head(10)
preview["Date"] = preview["Date"].dt.strftime("%b %Y")
st.dataframe(preview, use_container_width=True)

# full daily CSV download (more detail)
full_future = future_pred.copy()
full_future["Predicted"] = full_future["Predicted"].round(1)
csv = full_future.rename(columns={"Predicted":"Forecasted Sales"}).to_csv(index=False).encode("utf-8")
st.download_button("Download full forecast (daily CSV)", data=csv, file_name="future_forecast.csv", mime="text/csv")
st.caption("Preview shows monthly values; download for daily detail.")



##################   RELIABLITY AND NOTES   ##################

st.subheader("How reliable is this?")
st.markdown(f"""
- **Model**: LightGBM (gradient-boosted trees)  
- **Held-out accuracy (MAE)**: **{mae_text}** (lower = better)  
- **Training period**: **{hist['Date'].min().date()} → {hist['Date'].max().date()}**  
- **Main signals used**: recent sales trend, temperature, calendar patterns (month/weekday/weekend)  
- **Limitations**: promotions, stockouts, holidays, and external events are not included yet.  
- **Rule of thumb**: forecasts are more reliable in the **next 1–2 months**; uncertainty grows further out.
""")
