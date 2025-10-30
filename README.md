# Sales Forecast Dashboard

An interactive **Streamlit web application** for forecasting and analyzing sales trends using **machine learning (LightGBM)**.  
This dashboard transforms historical sales data into clear insights. This is helping users understand seasonal trends, weekday performance, and future sales projections.

---

## Project Overview

This project predicts future sales based on:
- **Historical sales data**
- **Temperature trends**
- **Calendar patterns** (month, weekday, weekends)

The goal is to make forecasting **simple, interpretable, and visually engaging** for decision-makers and data-driven teams.

---

## Key Features

### Forecasting
- Predicts daily or monthly sales for up to **6 months ahead**
- Automatically incorporates recent temperature trends
- Confidence intervals visualize forecast uncertainty

### Visual Insights
- **Sales vs. Forecast Plot:** Blue = actuals, orange = forecast  
- **Weekday Activity Index:** Shows which weekdays outperform the average  
- **Trend Direction Indicator:** Detects if sales are rising, falling, or stable  
- **Feature Importance:** Highlights what factors drive the model’s predictions  

### Data Export
- Download full forecast results (daily or monthly) as CSV files.

---

## Tech Stack

- **Python 3.10+**
- **Streamlit** – interactive web app framework  
- **LightGBM** – machine learning model for forecasting  
- **Pandas / NumPy** – data processing  
- **Plotly** – rich, interactive visualizations  
- **Joblib** – model loading and saving  

---

## Folder Structure
Sales Forecast/
│
├── app/
│ ├── streamlit_app.py # Main Streamlit application
│
├── data/
│ └── processed/
│ ├── daily_sales.csv # Historical sales data
│ ├── features.csv # Engineered feature set
│ └── test_predictions.csv # Optional test data
│
├── models/
│ └── lightgbm_model.pkl # Trained LightGBM model
│
└── README.md

## How to Run the App
### 1. Clone the repository
git clone https://github.com/helendoo/sales-forecast-dashboard.git
cd sales-forecast-dashboard/app

### 2. Install dependencies
pip install -r requirements.txt

### 3. Run the Streamlit app
streamlit run streamlit_app.py
Then open the link shown in your terminal (usually http://localhost:8501)


