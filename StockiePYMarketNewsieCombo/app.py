import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import yfinance as yf
import seaborn as sns
import io
import edgar
import base64
import ta
import time as time
import requests
import os
import json 

from bs4 import BeautifulSoup
from io import BytesIO
from io import StringIO
from dotenv import load_dotenv
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
from fredapi import Fred 
from flask import Flask, render_template, request, redirect, url_for, jsonify

app = Flask(__name__)

search_history = []

# Index route main
@app.route("/", methods=["GET", "POST"])
def index():
    current_time = time.time()
    if request.method == 'POST':
        stock_name = request.form['stock_name']
        stock_data = get_stock_data(stock_name)
        # Get stock returns and generate the distribution plot
        period = request.form.get("period", "1y")
        returns = get_stock_returns(stock_name, period)
        distribution_plot_base64 = generate_distribution_plot(returns)
        ma_plot_base64 = generate_ma_plot(stock_data)
        


        # Calculate moving averages
        stock_data = yf.Ticker(stock_name).history(period='1y')
        stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
        stock_data['MA200'] = stock_data['Close'].rolling(window=200).mean()

        last_price = stock_data.iloc[-1]['Close']
        ma50 = stock_data.iloc[-1]['MA50']
        ma200 = stock_data.iloc[-1]['MA200']
        
        # Calculate technical indicators
        rsi = ta.momentum.RSIIndicator(stock_data['Close']).rsi().iloc[-1]
        macd_indicator = ta.trend.MACD(stock_data['Close'])
        macd = ta.trend.MACD(stock_data['Close']).macd().iloc[-1]
        macd_signal = macd_indicator.macd_signal().iloc[-1]
        bb = ta.volatility.BollingerBands(stock_data['Close'])
        bb_upper = bb.bollinger_hband().iloc[-1]
        bb_middle = bb.bollinger_mavg().iloc[-1]
        bb_lower = bb.bollinger_lband().iloc[-1]
        stoch = ta.momentum.StochasticOscillator(stock_data['High'], stock_data['Low'], stock_data['Close'])
        stochastic_k = stoch.stoch().iloc[-1]
        stochastic_d = stoch.stoch_signal().iloc[-1]
        
        
        
        success = macd > macd_signal

        search_history.append({
            "stock_name": stock_name,
            "period": period,
            "trend": "Bullish" if success else "Bearish"
    })

        # Get earnings and revenue growth
        #earnings_revenue_growth = get_earnings_revenue_growth(stock_name)
        
        #Generate Summary
        summary_text = generate_summary(stock_name, last_price, ma50, ma200, rsi, macd, macd_signal)
        
        # Get earnings and revenue growth
        earnings_revenue_growth = get_earnings_revenue_growth(stock_name)

        return render_template('index.html', stock_name=stock_name, last_price=last_price, ma50=ma50, ma200=ma200, success=success, distribution_plot=distribution_plot_base64, ma_plot=ma_plot_base64, rsi=rsi, macd=macd, bb_upper=bb_upper, bb_middle=bb_middle, bb_lower=bb_lower, stochastic_k=stochastic_k, stochastic_d=stochastic_d, summary_text=summary_text, earnings_revenue_growth=earnings_revenue_growth, search_history=search_history)
    else:
        return render_template('index.html', search_history=search_history)

# Generate Stock Data
def get_stock_data(ticker, period='1y'):
    stock = yf.Ticker(ticker)
    stock_history = stock.history(period=period)
    if stock_history.empty:
        return None
    stock_history['Returns'] = stock_history['Close'].pct_change()
    stock_history['MA50'] = stock_history['Close'].rolling(window=50).mean()
    stock_history['MA200'] = stock_history['Close'].rolling(window=200).mean()
    return stock_history

# Get stock returns function
def get_stock_returns(ticker, period):
    stock = yf.Ticker(ticker)
    stock_history = stock.history(period=period)
    stock_history['Returns'] = stock_history['Close'].pct_change()
    return stock_history['Returns'].dropna()

# Generate distribution plot function
def generate_distribution_plot(returns):
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

# Generate MA Plot Function
def generate_ma_plot(stock_data):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=stock_data[['Close', 'MA50', 'MA200']])
    plt.title("Stock Price with 50-day and 200-day Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price")
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    plot_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    return plot_base64


def generate_summary(stock_name, last_price, ma50, ma200, rsi, macd, macd_signal):
    summary = []

    if ma50 > ma200:
        summary.append(f"{stock_name} has a positive trend, with the 50-day moving average being higher than the 200-day moving average.")
    else:
        summary.append(f"{stock_name} has a negative trend, with the 50-day moving average being lower than the 200-day moving average.")
    
    if rsi < 30:
        summary.append(f"The RSI is {rsi}, indicating that the stock may be oversold.")
    elif rsi > 70:
        summary.append(f"The RSI is {rsi}, indicating that the stock may be overbought.")
    else:
        summary.append(f"The RSI is {rsi}, indicating that the stock is currently neither overbought nor oversold.")
       
    if macd > macd_signal:
        summary.append(f"The MACD is {macd:.2f} and above the signal line ({macd_signal:.2f}), suggesting a bullish trend.")
    else:
        summary.append(f"The MACD is {macd:.2f} and below the signal line ({macd_signal:.2f}), suggesting a bearish trend.")


    return " ".join(summary)

# Load environment variables from .env file
load_dotenv()

ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
FRED_API_KEY = os.getenv('FRED_API_KEY')

fred = Fred(api_key=FRED_API_KEY)
ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')



# Get earnings and revenue growth function
def get_earnings_revenue_growth(stock_name, years=5, api_key='ALPHA_VANTAGE_API_KEY'):
    try:
        url = f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={stock_name}&apikey={api_key}'
        response = requests.get(url)
        data = json.loads(response.text)

        if 'annualReports' in data:
            annual_reports = data['annualReports']

            growth_data = {}
            prev_earnings = None
            prev_revenue = None

            for report in annual_reports[:years]:
                year = report['fiscalDateEnding'][:4]
                earnings = float(report['netIncome'])
                revenue = float(report['totalRevenue'])

                if prev_earnings is not None and prev_revenue is not None:
                    growth_data[year] = {
                        'earnings': ((earnings - prev_earnings) / prev_earnings) * 100,
                        'revenue': ((revenue - prev_revenue) / prev_revenue) * 100,
                    }

                prev_earnings = earnings
                prev_revenue = revenue

            return growth_data
        else:
            return None
    except Exception as e:
        print(f"Error fetching financials: {str(e)}")
        return None
    
@app.route('/MarketNewsie')
async def MarketNewsie():
    news = await fetch_and_parse_news()
    return render_template('marketnewsie.html')

async def fetch_and_parse_news():
    all_news = []

    for source in sources:
        try:
            news_items = source["parse"]()
            all_news.append({"sourceName": source["name"], "newsItems": news_items})
        except Exception as error:
            print(f"Failed to fetch news from {source['name']}:", error)

    return all_news

# Add the functions to parse each news source

def parse_federal_reserve():
    url = 'https://www.federalreserve.gov/newsevents/pressreleases.htm'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    news_items = []

    elements = soup.select('.panel-body ul.list-unstyled.panel-body__list a')
    for el in elements:
        news_items.append({
            'title': el.text.strip(),
            'url': f'https://www.federalreserve.gov{el["href"]}',
        })

    return news_items

def parse_yahoo_finance():
    url = 'https://finance.yahoo.com/news'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    news_items = []

    elements = soup.select('.js-stream-content a')
    for el in elements:
        news_items.append({
            'title': el.text.strip(),
            'url': el['href'] if el['href'].startswith('http') else f'https://finance.yahoo.com{el["href"]}',
        })

    return news_items

def parse_reuters():
    url = 'https://www.reuters.com/world/'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    news_items = []

    elements = soup.select('ul.story-collection__list__2M49i li')
    for el in elements:
        title_element = el.find('a.text__heading_3_story_card_hero__ISKN9, a.text__heading_4_story_card_small__2vDm1')
        title = title_element.text.strip()
        href = title_element['href']

        if href:
            absolute_url = href if href.startswith('http') else f'https://www.reuters.com{href}'
            news_items.append({
                'title': title,
                'url': absolute_url,
            })

    return news_items

sources = [
    {
        "name": "Federal Reserve",
        "url": "https://www.federalreserve.gov/newsevents/pressreleases.htm",
        "parse": parse_federal_reserve,
    },
    {
        "name": "Yahoo Finance",
        "url": "https://finance.yahoo.com/news",
        "parse": parse_yahoo_finance,
    },
    {
        "name": "Reuters",
        "url": "https://www.reuters.com/world/",
        "parse": parse_reuters,
    },
]


if __name__ == "__main__":
    app.run(debug=True)
