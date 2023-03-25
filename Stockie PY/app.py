import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import yfinance as yf
import io
from io import BytesIO
import base64
import ta
import time as time

from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Index route
@app.route("/", methods=["GET", "POST"])
def index():
    current_time = time.time()
    if request.method == 'POST':
        stock_name = request.form['stock_name']
        # Get stock returns and generate the distribution plot
        period = request.form.get("period", "1y")
        returns = get_stock_returns(stock_name, period)
        plot_base64 = generate_plot(returns)

        # Calculate moving averages
        stock_data = yf.Ticker(stock_name).history(period='1y')
        stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
        stock_data['MA200'] = stock_data['Close'].rolling(window=200).mean()

        last_price = stock_data.iloc[-1]['Close']
        ma50 = stock_data.iloc[-1]['MA50']
        ma200 = stock_data.iloc[-1]['MA200']

        success = stock_data['MA50'].iloc[-1] > stock_data['MA200'].iloc[-1]

        # Calculate technical indicators
        rsi = ta.momentum.RSIIndicator(stock_data['Close']).rsi().iloc[-1]
        macd = ta.trend.MACD(stock_data['Close']).macd().iloc[-1]
        bb = ta.volatility.BollingerBands(stock_data['Close'])
        bb_upper = bb.bollinger_hband().iloc[-1]
        bb_middle = bb.bollinger_mavg().iloc[-1]
        bb_lower = bb.bollinger_lband().iloc[-1]
        stoch = ta.momentum.StochasticOscillator(stock_data['High'], stock_data['Low'], stock_data['Close'])
        stochastic_k = stoch.stoch().iloc[-1]
        stochastic_d = stoch.stoch_signal().iloc[-1]

        # Get earnings and revenue growth
        earnings_revenue_growth = get_earnings_revenue_growth(stock_name)

        return render_template('index.html', stock_name=stock_name, last_price=last_price, ma50=ma50, ma200=ma200, success=success, plot=plot_base64, rsi=rsi, macd=macd, bb_upper=bb_upper, bb_middle=bb_middle, bb_lower=bb_lower, stochastic_k=stochastic_k, stochastic_d=stochastic_d, earnings_revenue_growth=earnings_revenue_growth)
    else:
        return render_template('index.html')

# Get stock returns function
def get_stock_returns(ticker, period):
    stock = yf.Ticker(ticker)
    stock_history = stock.history(period=period)
    stock_history['Returns'] = stock_history['Close'].pct_change()
    return stock_history['Returns'].dropna()

# Generate plot function
def generate_plot(returns):
    plt.hist(returns, bins=30, density=True, alpha=0.6, color="b")
    plt.xlabel("Returns")
    plt.ylabel("Frequency")
    plt.title("Stock Returns Distribution Plot")

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()

    plot_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()

    return plot_base64

# Get earnings and revenue growth function
def get_earnings_revenue_growth(stock_name):
    stock = yf.Ticker(stock_name)
    retries = 3
    delay = 5  # 5 seconds delay

    for i in range(retries):
        try:
            financials = stock.financials.T
            break
        except Exception as e:
            if i < retries - 1:
                print(f"Error fetching financials. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Error fetching financials. Aborting.")
                return {}
    financials = stock.financials.T
    financials['Earnings Growth'] = financials['Gross Profit'].pct_change() * 100
    financials['Revenue Growth'] = financials['Total Revenue'].pct_change() * 100

    growth_data = {}
    for index, row in financials.iterrows():
        year = index.year
        growth_data[year] = {
            'earnings': row['Earnings Growth'],
            'revenue': row['Revenue Growth']
        }

    return growth_data

if __name__ == "__main__":
    app.run(debug=True)
