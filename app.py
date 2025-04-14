# === Import modules ===
# Data source
import yfinance as yf
from functions import *

# Data processing
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Feature engineering / technical analysis
from ta.volatility import BollingerBands
from ta.trend import MACD, SMAIndicator
from ta.momentum import RSIIndicator

# Modeling
import xgboost as xgb

# Model performance
from sklearn.metrics import classification_report, confusion_matrix

# Dashboard
import panel as pn
import hvplot.pandas


# Settings / other
import warnings
from datetime import date, timedelta

# === Config ===

warnings.simplefilter('ignore')

# === Get stock data ===

tickers = ['ALB', 'SQM', 'PLL']  # We drop 'LAC'
start_date = "2010-01-01"
today = date.today()
# Go back to the last Sunday
days_since_sunday = (today.weekday() + 1) % 7
last_sunday = today - timedelta(days=days_since_sunday)
end_date = last_sunday.strftime("%Y-%m-%d")

# Download weekly adjusted close prices
price_df = get_weekly_data(tickers, start_date, end_date)
# Calculate weekly returns for each ticker
returns_df = price_df.pct_change().dropna()


# Create lithium sentiment index
# Simple average across all 3 tickers.
# This line creates a synthetic sentiment index that rises
# or falls depending on the average performance of the stocks.
returns_df['Lithium_Index'] = returns_df.mean(axis=1)
# Convert sentiment into Bullish/Bearish labels (for classification)
returns_df['Sentiment_Label'] = returns_df['Lithium_Index'].apply(
    lambda x: 1 if x > 0 else 0)

# Turn returns into a Lithium market index level
# Start index at 100 (like many indices do)
returns_df['Lithium_Market_Index'] = (
    1 + returns_df['Lithium_Index']).cumprod() * 100

# === Merge market data with historical events

"""
* Adding a simple binary `Has_Event` column based on whether there's
a historic lithium event on a given day — treating multiple events
on the same date as just "1"
* Aligning the event dates to the nearest valid trading day in your
DataFrame. Here's a clean way to do that
"""
# Your original DataFrame
# Ensure datetime index is sorted
returns_df.index = pd.to_datetime(returns_df.index)
returns_df.sort_index(inplace=True)

# Parse event dates
raw_event_dates = pd.to_datetime(list(events().keys()))

# Snap each event to the next available
# trading date (or last one if out of range)
indices = returns_df.index.searchsorted(raw_event_dates)
# prevent out-of-bounds
indices = np.minimum(indices, len(returns_df.index) - 1)

# Get actual dates from the index
valid_event_dates = returns_df.index[indices]

# Mark them in the DataFrame
returns_df["Has_Event"] = returns_df.index.isin(valid_event_dates).astype(int)

# === Feature engineering ===
# We’ll apply indicators to the Lithium_Index (avg weekly return)
# Alternatively you can apply them to each ticker separately

# Example: SMA and RSI on Lithium_Index
sma = SMAIndicator(close=returns_df["Lithium_Index"], window=4)
rsi = RSIIndicator(close=returns_df["Lithium_Index"], window=4)

returns_df["SMA_4"] = sma.sma_indicator()
returns_df["RSI_4"] = rsi.rsi()

# MACD
macd = MACD(close=returns_df["Lithium_Index"])
returns_df["MACD"] = macd.macd()
returns_df["MACD_Signal"] = macd.macd_signal()

# Bollinger Bands (for volatility)
bb = BollingerBands(close=returns_df["Lithium_Index"])
returns_df["BB_High"] = bb.bollinger_hband()
returns_df["BB_Low"] = bb.bollinger_lband()
returns_df["BB_Width"] = returns_df["BB_High"] - returns_df["BB_Low"]

# === Modeling ===

model_df = returns_df.dropna().copy().reset_index().rename(
    columns={"index": "Date"})
model_df["Date"] = pd.to_datetime(model_df["Date"])

# Define Features & target
feature_cols = [
    "Has_Event", "SMA_4", "RSI_4",
    "MACD", "MACD_Signal", "BB_Width"]

# Spllit dataset (time-wise)
split_index = int(len(model_df) * 0.8)
train_df = model_df.iloc[:split_index]
test_df = model_df.iloc[split_index:]

X_train = train_df[feature_cols]
y_train = train_df["Sentiment_Label"]
X_test = test_df[feature_cols]
y_test = test_df["Sentiment_Label"]

# Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model
model = xgb.XGBClassifier(
    objective="binary:logistic",
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Predict on Test Set Only
y_pred = model.predict(X_test_scaled)

# Add Predictions to model_df (only for test part)
model_df["Predicted_Label"] = np.nan  # initialize with NaNs
model_df.loc[split_index:, "Predicted_Label"] = y_pred

# Predict probabilities (confidence that label is bullish)
# Probability of class 1 (bullish)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# Store in the test portion of model_df
model_df.loc[split_index:, "Confidence_Score"] = y_proba

# Mark correctness
model_df["Correct"] = np.nan
model_df.loc[split_index:, "Correct"] = (
    model_df.loc[split_index:, "Predicted_Label"]
    == model_df.loc[split_index:, "Sentiment_Label"])

# === Widgets and Callbacks ===

# Sidebar buttons
button_1 = pn.widgets.Button(
    name="Predicted Sentiment", button_type="primary",
    button_style="outline", icon="", styles={'width': '90%'})
button_2 = pn.widgets.Button(
    name="Info", button_type="primary",
    button_style="outline", icon="", styles={'width': '90%'})
button_3 = pn.widgets.Button(
    name="Graphs", button_type="primary",
    button_style="outline", icon="", styles={'width': '90%'})
button_4 = pn.widgets.Button(
    name="Model Performance", button_type="primary",
    button_style="outline", icon="", styles={'width': '90%'})
feature_cols = [
        "Has_Event", "SMA_4", "RSI_4",
        "MACD", "MACD_Signal", "BB_Width"]
X_next = returns_df.dropna().copy().tail(1)[feature_cols]
X_next_scaled = scaler.transform(X_next)
prediction_date = model_df.dropna().tail(1)['Date'].values[0]
next_week = pd.to_datetime(prediction_date) + pd.Timedelta(days=7)
next_label = model.predict(X_next_scaled)[0]
label_str = "📈 Bullish" if next_label == 1 else "📉 Bearish"
next_proba = model.predict_proba(X_next_scaled)[0][1]
confidence_str = f"{next_proba:.2%}"
color = "yellow" if next_label == 1 else "crimson"


# creating page content

mapping = {
    "Page1": create_page_1(color, next_week, label_str, confidence_str, disclaimer),
    "Page2": create_page_2(),
    "Page3": create_page_3(returns_df, model_df, disclaimer),
    "Page4": create_page_4(y_test, y_pred, disclaimer)
}

# === Layout ===

# sidebar
sidebar = pn.Column(pn.pane.Markdown("## 🗂️ Pages"), button_1, button_2, button_3, button_4,
					styles={"width": "100%", "padding": "15px"})

# main area
main_area = pn.Column(mapping["Page1"], styles={"width":"100%"})

# Show page
def show_page(page_key):
    """
    callback function for all buttons
    """
    main_area.clear()  # It will clear the main area each time function is called
    main_area.append(mapping[page_key])  # and the new page will be appended
    
# calling actions when buttons is clicked
button_1.on_click(lambda event: show_page("Page1"))
button_2.on_click(lambda event: show_page("Page2"))
button_3.on_click(lambda event: show_page("Page3"))
button_4.on_click(lambda event: show_page("Page4"))

# App layout
template = pn.template.BootstrapTemplate(
    title="Lithium Market Sentiment Dashboard",
    sidebar=[sidebar],
    main=[main_area],
    header_background="black", 
    # site="CoderzColumn", logo="cc.png",
    theme=pn.template.DarkTheme,
    sidebar_width=250, ## Default is 330
    busy_indicator=None,
)

# === Run dashboard ===
template.servable()
