import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from yahoo_fin import stock_info as si
import time
import threading

# Fetch all penny stocks from Yahoo Finance
def fetch_all_penny_stocks():
    stock_list = si.tickers_nse()
    penny_stocks = []
    
    for stock in stock_list:
        try:
            price = si.get_live_price(stock + ".NS")
            if price < 50:
                penny_stocks.append(stock)
        except:
            continue  # Skip if stock data is unavailable
    
    return penny_stocks

def fetch_stock_data(ticker):
    df = si.get_data(ticker + ".NS")
    df = df[['open', 'high', 'low', 'close', 'volume']]
    df.rename(columns={'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    df.dropna(inplace=True)
    return df

# Preprocess data
def preprocess_data(df):
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['Volatility'] = df['Close'].pct_change().rolling(10).std()
    df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(14).mean()))
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['ATR'] = df['High'] - df['Low']
    df['Bollinger_Upper'] = df['MA_10'] + (df['Close'].rolling(window=10).std() * 2)
    df['Bollinger_Lower'] = df['MA_10'] - (df['Close'].rolling(window=10).std() * 2)
    df.dropna(inplace=True)
    return df

# Identify high-earning penny stocks
def find_high_earning_penny_stocks():
    penny_stocks = fetch_all_penny_stocks()
    high_earning_stocks = []
    
    for stock in penny_stocks:
        df = fetch_stock_data(stock)
        df = preprocess_data(df)
        
        if df['Volatility'].iloc[-1] > df['Volatility'].mean() and df['RSI'].iloc[-1] > 50:
            buy_price = df['Close'].iloc[-1]
            risk_factor = df['Volatility'].iloc[-1] * 1.5
            stop_loss = buy_price - (buy_price * risk_factor)
            target_price = buy_price + (buy_price * (risk_factor * 2))
            high_earning_stocks.append((stock, buy_price, stop_loss, target_price))
    return high_earning_stocks

# Streamlit Web App
st.title("Penny Stock Scanner & Predictor")
high_earning_penny_stocks = find_high_earning_penny_stocks()
for stock, buy_price, stop_loss, target_price in high_earning_penny_stocks:
    st.write(f'**Penny Stock:** {stock}')
    st.write(f'Buy at: ₹{buy_price:.2f} | Stop-Loss: ₹{stop_loss:.2f} | Target: ₹{target_price:.2f}')
    st.write("---")

# Notification Section in Streamlit
st.subheader("Live Stock Alerts")
alert_placeholder = st.empty()

def real_time_stock_monitor(interval=60):
    while True:
        high_earning_penny_stocks = find_high_earning_penny_stocks()
        alert_text = ""
        for stock, buy_price, stop_loss, target_price in high_earning_penny_stocks:
            alert_text += f'**{stock}** - Buy at: ₹{buy_price:.2f}, Target: ₹{target_price:.2f}, Stop-Loss: ₹{stop_loss:.2f}\n'
        alert_placeholder.markdown(alert_text)
        time.sleep(interval)

# Run real-time monitoring in a separate thread
monitor_thread = threading.Thread(target=real_time_stock_monitor, daemon=True)
monitor_thread.start()
st.write("Real-time monitoring enabled. Checking stocks every 60 seconds...")
