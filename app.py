import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import nsetools
from nsetools import Nse
import time

# Fetch stock data from NSE
nse = Nse()

def fetch_all_penny_stocks():
    stock_codes = nse.get_stock_codes()
    penny_stocks = []
    
    for code in stock_codes:
        if code == 'SYMBOL':  # Skip header entry
            continue
        try:
            stock_data = nse.get_quote(code)
            if stock_data and stock_data['lastPrice'] < 50:
                penny_stocks.append(code)
        except:
            continue  # Skip stocks with errors
    
    return penny_stocks

def fetch_stock_data(ticker):
    stock_data = nse.get_quote(ticker)
    df = pd.DataFrame([stock_data])
    df['Date'] = pd.to_datetime(df['lastUpdateTime'])
    df.set_index('Date', inplace=True)
    return df

# Preprocess data
def preprocess_data(df):
    df = df[['open', 'dayHigh', 'dayLow', 'lastPrice', 'quantityTraded']]
    df.rename(columns={'dayHigh': 'High', 'dayLow': 'Low', 'lastPrice': 'Close', 'quantityTraded': 'Volume'}, inplace=True)
    df.dropna(inplace=True)
    
    # Creating new features
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['Volatility'] = df['Close'].pct_change().rolling(10).std()
    df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(14).mean()))
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['ATR'] = df['High'] - df['Low']
    df['Bollinger_Upper'] = df['MA_10'] + (df['Close'].rolling(window=10).std() * 2)
    df['Bollinger_Lower'] = df['MA_10'] - (df['Close'].rolling(window=10).std() * 2)
    
    # Handling missing values after rolling computations
    df.dropna(inplace=True)
    
    return df

# Identify high-earning penny stocks with dynamic buy/sell triggers
def find_high_earning_penny_stocks():
    penny_stocks = fetch_all_penny_stocks()
    high_earning_stocks = []
    
    for stock in penny_stocks:
        df = fetch_stock_data(stock)
        df = preprocess_data(df)
        
        if df['Volatility'].iloc[-1] > df['Volatility'].mean() and df['RSI'].iloc[-1] > 50:
            buy_price = df['Close'].iloc[-1]
            
            # Dynamic Stop-Loss & Target Adjustments
            risk_factor = df['Volatility'].iloc[-1] * 1.5  # Higher volatility → wider stop-loss
            stop_loss = buy_price - (buy_price * risk_factor)
            target_price = buy_price + (buy_price * (risk_factor * 2))  # Reward is twice the risk
            
            buy_date = df.index[-1]
            high_earning_stocks.append((stock, buy_price, stop_loss, target_price, buy_date))
    return high_earning_stocks

# Streamlit Web App
st.title("Penny Stock Scanner & Predictor")
high_earning_penny_stocks = find_high_earning_penny_stocks()
for stock, buy_price, stop_loss, target_price, buy_date in high_earning_penny_stocks:
    st.write(f'**Penny Stock:** {stock}')
    st.write(f'Buy at: ₹{buy_price:.2f} | Stop-Loss: ₹{stop_loss:.2f} | Target: ₹{target_price:.2f} | Date: {buy_date}')
    st.write("---")

# Load data for a specific stock
data = fetch_stock_data(high_earning_penny_stocks[0][0])
data = preprocess_data(data)

# Splitting data into training and testing
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[['Close', 'MA_10', 'MA_50', 'Volatility', 'RSI', 'MACD', 'ATR']])
X = data_scaled[:-1]  # Features
Y = data_scaled[1:, 0]  # Target (Next Day Close Price)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train a RandomForest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

# Predict next day's price
predicted_prices = model.predict(X_test)

# Visualize Predictions
st.subheader("Stock Price Prediction vs Actual")
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(Y_test, label='Actual Prices', color='blue')
ax.plot(predicted_prices, label='Predicted Prices', color='red')
ax.legend()
ax.set_title('Stock Price Prediction vs Actual')
ax.set_xlabel('Time')
ax.set_ylabel('Price')
st.pyplot(fig)
