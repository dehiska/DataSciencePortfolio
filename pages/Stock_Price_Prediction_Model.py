import streamlit as st
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

# --- Page Configuration ---
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

st.title("📈 Stock Price Prediction using Random Forest")
st.markdown("""
This page demonstrates a machine learning model that predicts the next day's closing price
for several major stocks. The model uses historical closing prices and Simple Moving Averages (SMAs)
as features. The models are trained in real-time when you first visit the page.
""")

# --- Helper Functions with Caching ---

@st.cache_data(ttl=3600) # Cache data for 1 hour
def fetch_stock_data(tickers, api_key):
    """Fetches and processes stock data for a list of tickers."""
    stock_data = {}
    progress_bar = st.progress(0, text="Fetching data...")
    for i, ticker in enumerate(tickers):
        try:
            ts = TimeSeries(key=api_key, output_format='pandas')
            data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
            
            data.rename(columns={
                '1. open': 'Open', '2. high': 'High', '3. low': 'Low',
                '4. close': 'Close', '5. volume': 'Volume'
            }, inplace=True)
            
            # Feature Engineering
            df = data.copy()
            df['SMA_10'] = df['Close'].rolling(window=10).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['Target'] = df['Close'].shift(-1)
            df.dropna(inplace=True)
            stock_data[ticker] = df
            
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}. Check API key limits.")
            continue
        finally:
            # Update progress bar
            progress_bar.progress((i + 1) / len(tickers), text=f"Fetching data for {ticker}...")
    
    progress_bar.empty()
    return stock_data

@st.cache_resource # Cache the trained models to avoid retraining
def train_models(stock_data):
    """Trains a RandomForest model for each stock."""
    results = {}
    for ticker, df in stock_data.items():
        X = df[['Close', 'SMA_10', 'SMA_50']]
        y = df['Target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        results[ticker] = {
            "Model": model, "MSE": mse, "MAE": mae,
            "Predictions": (y_test, y_pred)
        }
    return results

# --- Main Page Logic ---

try:
    api_key = st.secrets["my_api_key"]
except FileNotFoundError:
    st.error("Secrets file not found. Please create a .streamlit/secrets.toml file with your API key.")
    st.code("my_api_key = 'YOUR_ALPHA_VANTAGE_API_KEY'")
    st.stop()
except KeyError:
    st.error("API key not found in secrets.toml. Please add it.")
    st.code("my_api_key = 'YOUR_ALPHA_VANTAGE_API_KEY'")
    st.stop()


tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

stock_data = fetch_stock_data(tickers, api_key)

if not stock_data:
    st.warning("Could not fetch data for any stocks. Please check your API key and connection.")
    st.stop()
    
with st.spinner("Training models... This may take a moment on first run."):
    results = train_models(stock_data)

st.success("Data loaded and models trained successfully!")

# --- Display Results and Predictions ---

st.header("Model Performance and Predictions")

col1, col2 = st.columns(2)
plot_columns = [col1, col2]

for i, ticker in enumerate(results.keys()):
    with plot_columns[i % 2]:
        st.subheader(f"{ticker} Stock Price Prediction")
        
        y_test, y_pred = results[ticker]['Predictions']
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(y_test.values, label="Actual Prices", alpha=0.8)
        ax.plot(y_pred, label="Predicted Prices", alpha=0.7, linestyle='--')
        ax.set_title(f"{ticker} Actual vs. Predicted Prices")
        ax.set_xlabel("Test Samples (Days)")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        st.pyplot(fig)
        
        st.metric(label="Mean Squared Error (MSE)", value=f"{results[ticker]['MSE']:.2f}")
        st.metric(label="Mean Absolute Error (MAE)", value=f"{results[ticker]['MAE']:.2f}")

st.header("🔮 Next Day's Predicted Closing Price")

pred_cols = st.columns(len(tickers))
for i, ticker in enumerate(results.keys()):
    with pred_cols[i]:
        model = results[ticker]["Model"]
        df = stock_data[ticker]
        latest_data = df[['Close', 'SMA_10', 'SMA_50']].iloc[-1].values.reshape(1, -1)
        predicted_price = model.predict(latest_data)
        st.metric(label=f"Predicted Close for {ticker}", value=f"${predicted_price[0]:.2f}")
