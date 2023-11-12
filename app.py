# This code block is importing the necessary libraries and modules that will be used in the rest of
# the code. These libraries include:
# Import necessary libraries
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from datetime import date
from datetime import timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import pytrends
from pytrends.request import TrendReq
import time
import investpy
from hijri_converter import Hijri, Gregorian, convert
from ta import add_all_ta_features
from prophet import Prophet

# The code block you provided is responsible for displaying the title and description of the stock
# market forecasting app.
# Title and description
app_name = 'Stock Market Forecasting App'
st.title(app_name)
st.subheader('This trial version is used to forecast Stock Market & Crypto Prices')
st.image("https://img.freepik.com/free-vector/gradient-stock-market-concept_23-2149166910.jpg")


# The code you provided is responsible for downloading and displaying the stock market data for a
# selected ticker symbol.
# Add Ticker Symbol List
import streamlit as st

sector_list = ['Semiconductor', 'Technology']
sector_selected = st.sidebar.selectbox('Select Sector', sector_list)

if sector_selected == 'Semiconductor':
    tickers = ["INTC", "NVDA", "AMD","QCOM","TSM"]
elif sector_selected == 'Technology':
    tickers = ['MSFT', 'GOOG', "APPL"]

ticker = st.sidebar.selectbox('Select Ticker', tickers)


# Calculate the date range for the last 2 years
today = date.today()
two_years_ago = today.replace(year=today.year - 2)

# Check if today is Saturday or Sunday
if today.weekday() == 5:  # Saturday
    end_date = today - timedelta(days=1)
elif today.weekday() == 6:  # Sunday
    end_date = today - timedelta(days=2)
else:
    end_date = today

two_years_ago = today.replace(year=today.year - 2)

# Download data
data = yf.download(ticker, start=two_years_ago, end=today)

data.insert(0, "Date", data.index, True)

# Show the downloaded data
st.write(f'Data for {ticker} from {two_years_ago} to {today}:')

# Display the downloaded data
st.write(data)

# This code is responsible for visualizing the stock market data. It creates a line plot using Plotly
# Express (`px.line`) to show the closing price of the stock over time. The x-axis represents the date
# and the y-axis represents the closing price. The title of the plot is set to "Closing Price of the
# Stock Price". Finally, the plot is displayed using `st.plotly_chart` from the Streamlit library.
#Data Visualization
st.header('Data Visualization')
st.subheader('Plot of the Data')
fig1 = px.line(data, x='Date', y=data.columns, title='Closing Price of the Stock Price')
st.plotly_chart(fig1)


# The code calculates the weekday effect on stock market trends. It calculates the percentage return
# for each day of the week and then calculates the average return for each weekday. It then determines
# the current day of the week and checks if the average return for that day is positive or negative.
# If the average return is positive, it displays a "Buy" signal, indicating that it may be a good day
# to invest in the stock. If the average return is negative, it displays a "Negative" signal,
# indicating that it may not be a good day to invest.
# Weeday Effect

# Calculate the percentage return
data['Percentage_Return'] = data['Close'].pct_change() * 100

# Calculate weekday averages
weekday_avg = data.groupby(data.index.day_name())['Percentage_Return'].mean()

# Sort the weekdays in chronological order
weekday_avg = weekday_avg.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])


# Get the current day of the week
current_day = datetime.today().strftime('%A')
# Use Friday data for Saturday and Sunday
if current_day in ['Saturday', 'Sunday']:
    weekday_avg.loc[current_day] = data[data.index.day_name() == 'Friday']['Percentage_Return'].mean()


st.header('1. Weekday Effect')
st.text('Weekday Effect: Stock market trend influenced by day of the week, e.g., higher returns on Fridays.')
# Print the current day
st.write(f"Current Day: {current_day}")

# Check if the percentage return for the current day is positive
if weekday_avg[current_day] > 0:
    st.write('Signal: Buy')
else:
    st.write('Signal: Negative')
    
    


# The above code is performing the following tasks:
# Convert datetime index to months
data['Month'] = data.index.month

# Calculate monthly returns
data['Monthly_Returns'] = data['Close'].pct_change()

# Group by month and calculate mean returns
monthly_returns = data.groupby('Month')['Monthly_Returns'].mean()

# Get current month number
now = datetime.now()
current_month = now.month

# Create a Streamlit app
st.title("2. Monthly Return Effect")

# Display current month
st.write(f"Current Month: {current_month}")

# Check if the percentage return for the current month is positive
if monthly_returns[current_month] > 0:
    st.write('Signal: Buy')
else:
    st.write('Signal: Sell')




    
# The code calculates the quarterly returns for a given stock and then calculates the average return
# for each quarter. It then determines the current quarter and checks if the average return for the
# current quarter is positive or negative. If the average return is positive, it displays a "Buy"
# signal, otherwise it displays a "Negative" signal. This is used to analyze the quarterly effect on
# stock market trends.
# Calculate quarterly returns

# Calculate the quarter for each data point
data['Quarter'] = data.index.to_period('Q')

# Calculate quarterly averages
quarter_avg = data.groupby(data['Quarter'])['Percentage_Return'].mean()

# Get the current quarter
current_quarter = pd.to_datetime(datetime.today()).to_period('Q')

st.header('3. Quarterly Effect')
st.text('Quarterly Effect: Stock market trend influenced by Q1,Q2,Q3,Q4 .')

# Print the current quarter
st.write(f"Current quarter: {current_quarter}")

# Check if the average return for the current quarter is positive
if quarter_avg.loc[current_quarter] > 0:
    st.write('Signal: Buy')
else:
    st.write('Signal: Negative')
    
    



# This code block calculates the average percentage return for each day of the month using the `data`
# DataFrame. It first creates a new column called 'Day' which contains the day of the month for each
# data point. Then, it groups the data by the 'Day' column and calculates the mean of the
# 'Percentage_Return' column for each day.

# Calculate the day of the month for each data point
data['Day'] = data.index.day

# Calculate daily averages
day_avg = data.groupby(data['Day'])['Percentage_Return'].mean()

# Get the current day of the month
current_day = datetime.today().day

st.header('4. Day of the Month Effect')
st.text('Day of the Month Effect: Stock market trend influenced by specific days of the month.')

# Print the current day of the month
st.write(f"Current day of the month: {current_day}")

# Check if the average return for the current day of the month is positive
if day_avg.loc[current_day] > 0:
    st.write('Signal: Buy')
else:
    st.write('Signal: Negative')





# This code block calculates the week number for each data point in the `data` DataFrame. It uses the
# `index.week` attribute to get the week number for each date in the index.

# Calculate the week number for each data point
data['Week_Number'] = data.index.week

# Calculate weekly averages
week_avg = data.groupby(data['Week_Number'])['Percentage_Return'].mean()

# Get the current week number
current_week = datetime.today().date().isocalendar()[1]

st.header('5. Week Number Returns Effect')
st.text('Week Number Returns Effect: Stock market trend influenced by the week number within a year.')

# Print the current week number
st.write(f"Current week number: {current_week}")

# Check if the average return for the current week number is positive
if week_avg.loc[current_week] > 0:
    st.write('Signal: Buy')
else:
    st.write('Signal: Negative')


# Function to calculate Islamic month from Gregorian date
    """
    The code calculates the Islamic month from a given Gregorian date and uses it to determine the
    average percentage return for each Islamic month, displaying the results in a Streamlit app and
    providing a buy or sell signal based on the current Islamic month's return.
    
    :param date: The `date` parameter is the Gregorian date for which you want to calculate the
    corresponding Islamic month
    :return: The code is returning the Islamic month returns, which is the average percentage return for
    each Islamic month. It is also returning the current Islamic month and checking if the percentage
    return for the current month is positive or negative, and providing a signal to either buy or sell
    based on that. If there is no data available for the current month, it will display a message
    indicating that.
    """
def gregorian_to_islamic(date):
    hijri_date = convert.Gregorian(date.year, date.month, date.day).to_hijri()
    return hijri_date.month_name('en')


# Convert dates to Islamic months
data['Hijri_Months_EN'] = data.index.map(gregorian_to_islamic)

# Calculate Islamic month returns
Hijri_Months_Returns = data.groupby('Hijri_Months_EN')['Percentage_Return'].mean()

# Create a Streamlit app
st.title("Islamic Month Returns Effect")

# Display Islamic month returns
st.write(Hijri_Months_Returns)

# Get current Islamic month
now = pd.to_datetime(date.today())
current_month = gregorian_to_islamic(now)

# Display current Islamic month
st.write(f"Current Islamic Month: {current_month}")

# Check if the percentage return for the current Islamic month is positive
if current_month in Hijri_Months_Returns.index:
    if Hijri_Months_Returns[current_month] > 0:
        st.write('Signal: Buy')
    else:
        st.write('Signal: Sell')
else:
    st.write(f"No data available for {current_month}.")






# The code block you provided is calculating the "Turn of the Month Effect" on stock market trends. It
# checks if the current day is within a specified range around the turn of the month (e.g., 2 days
# before or after the last day of the month). If it is within the specified range, it calculates the
# average percentage return for those days and determines if it is positive or negative.
days_around_turn = 2

# Get the current day of the month
current_day = datetime.today().day

# Get the last day of the current month
last_day_of_month = (datetime.today().replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)

# Check if today is within the specified range around the turn of the month
if current_day <= days_around_turn or current_day >= (last_day_of_month.day - days_around_turn + 1):
    # Calculate the turn of the month signal
    turn_of_month_avg = data['Percentage_Return'][
        (data.index.day <= days_around_turn) |
        (data.index.day >= (last_day_of_month.day - days_around_turn + 1))
    ].mean()

    st.header('6. Turn of the Month Effect')
    st.text('Turn of the Month Effect: Stock market trend influenced by the beginning or end of each month.')

    # Print the turn of the month signal
    st.write(f"Turn of the Month Signal: {turn_of_month_avg}")

    # Check if the average return around the turn of the month is positive
    if turn_of_month_avg > 0:
        st.write('Signal: Buy')
    else:
        st.write('Signal: Negative')
else:
    st.header('5. Turn of the Month Effect')
    st.text('Turn of the Month Effect: Stock market trend influenced by the beginning or end of each month.')
    st.write('Signal: Neutral')












# This code block is calculating the FOMC (Federal Open Market Committee) announcement drift effect on
# stock market trends.
# Define FOMC announcement dates
fomc_dates = [
    '2023-01-25', '2023-03-15', '2023-05-03', '2023-06-14', '2023-07-26',
    '2023-09-20', '2023-11-01', '2023-12-13'
]

# Convert FOMC dates to datetime objects
fomc_dates = [datetime.strptime(date, '%Y-%m-%d').date() for date in fomc_dates]

# Define the number of days around the FOMC announcement date (e.g., 2 days)
days_around_fomc = 2

# Get the current date
current_date = datetime.today().date()

# Check if today is within the specified range around a FOMC announcement date
if any(abs((current_date - fomc_date).days) <= days_around_fomc for fomc_date in fomc_dates):
    # Calculate the FOMC announcement drift signal
    fomc_drift_avg = data['Percentage_Return'][
        (data.index >= (current_date - timedelta(days=days_around_fomc))) &
        (data.index <= (current_date + timedelta(days=days_around_fomc)))
    ].mean()

    st.header('7. FOMC Announcement Drift Effect')
    st.text('FOMC Announcement Drift Effect: Stock market trend influenced by FOMC announcement dates.')

    # Print the FOMC announcement drift signal
    st.write(f"FOMC Announcement Drift Signal: {fomc_drift_avg}")

    # Check if the average return around the FOMC announcement date is positive
    if fomc_drift_avg > 0:
        st.write('Signal: Buy')
    else:
        st.write('Signal: Negative')
else:
    st.header('6. FOMC Announcement Drift Effect')
    st.text('FOMC Announcement Drift Effect: Stock market trend influenced by FOMC announcement dates.')
    st.write('Signal: Neutral')





# The code block you provided is using the `pytrends` library to retrieve and plot Google Trends data
# for a specific ticker symbol.
# Set up pytrends
pytrends = TrendReq(hl='en-US', tz=360)

# Define the keyword (ticker) you want to search for
kw_list = [ticker]

# Build the payload for the API request
pytrends.build_payload(kw_list, timeframe='today 1-m', geo='US')

# Get the interest over time data
interest_over_time_df = pytrends.interest_over_time()

# Set up Streamlit
st.title("8. Google Trends Ticker Plotter")

# Plot the Google Trends data
st.header(f'Google Trends for {ticker}')
st.line_chart(interest_over_time_df[[ticker]])







# The above code is fetching historical stock data for a selected sector and its peer stocks, and then
# calculating the correlation matrix for the closing prices of these stocks. It then displays the
# correlation heatmap using the seaborn library in Streamlit.
peer_stock_symbols = tickers
# Fetch historical data for all peer stocks and Tesla
stock_data = {}
for symbol in [sector_selected] + peer_stock_symbols:
    stock = yf.Ticker(symbol)
    data = stock.history(period="1y")
    stock_data[symbol] = data['Close']

# Create a DataFrame with the stock data
df = pd.DataFrame(stock_data)

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Display the heatmap in Streamlit
st.header("9. Correlation Heatmap of Peer Stocks")

# Optionally, you can display the heatmap as an image
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Peer Stocks")

# Pass the Matplotlib figure object to st.pyplot()
st.pyplot(plt)






# The above code is a Python script that compares the cumulative returns of a stock (specified by the
# variable `ticker_stock`) with the cumulative returns of the S&P 500 index (specified by the variable
# `ticker_index`).
# Define tickers
ticker_stock = ticker
ticker_index = "^GSPC"

# Define date range (last 1 year)
end_date = date.today()
start_date = end_date - timedelta(days=365)

# Download data
data_stock = yf.download(ticker_stock, start=start_date, end=end_date)
data_index = yf.download(ticker_index, start=start_date, end=end_date)

# Calculate cumulative returns
data_stock['Cumulative_Return'] = (1 + data_stock['Close'].pct_change()).cumprod() - 1
data_index['Cumulative_Return'] = (1 + data_index['Close'].pct_change()).cumprod() - 1

# Create a Streamlit app
st.title("Comparisoion with S&P 500(Benchmark)")

# Create a line chart to compare cumulative returns using Plotly
fig = go.Figure()

# Add traces for stock and S&P 500 cumulative returns
fig.add_trace(go.Scatter(x=data_stock.index, y=data_stock['Cumulative_Return'], mode='lines', name=f'{ticker_stock} Cumulative Returns'))
fig.add_trace(go.Scatter(x=data_index.index, y=data_index['Cumulative_Return'], mode='lines', name='S&P 500 Cumulative Returns'))

# Update layout
fig.update_layout(title='Cumulative Returns Comparison',
                  xaxis_title='Date',
                  yaxis_title='Cumulative Returns')

# Display the line chart
st.plotly_chart(fig)


# The above code is adding technical indicators, including the Relative Strength Index (RSI), to a
# dataset called "data". It then defines RSI thresholds for buy and sell signals.

# Add technical indicators including RSI
data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

# Define RSI thresholds for buy and sell signals
rsi_buy_threshold = 30
rsi_sell_threshold = 70

# Create a Streamlit app
st.title("RSI Signal App")
st.subheader('Low RSI levels, below 30, generate buy signals and indicate an oversold or undervalued condition. High RSI levels, above 70, generate sell signals and suggest that a security is overbought or overvalued. ')
# Display RSI chart
st.line_chart(data['momentum_rsi'])

# Determine signals based on RSI
latest_rsi = data['momentum_rsi'].iloc[-1]

if latest_rsi < rsi_buy_threshold:
    signal = "Buy"
elif latest_rsi > rsi_sell_threshold:
    signal = "Sell"
else:
    signal = "Neutral"

# Display the signal
st.write(f"Latest RSI: {latest_rsi}")
st.write(f"Signal: {signal}")






# The above code is creating a Streamlit app that displays the Stochastic Oscillator signal for a
# given dataset. It defines the buy and sell thresholds for the Stochastic Oscillator, displays the
# data and chart, and determines the signal based on the latest Stochastic Oscillator value. The
# signal is then displayed on the app.
# Define Stochastic Oscillator thresholds for buy and sell signals
stochastic_buy_threshold = 20
stochastic_sell_threshold = 80

# Create a Streamlit app
st.title("Stochastic Oscillator Signal App")
st.subheader('The Stochastic Oscillator is a momentum indicator that shows the location of the close relative to the high-low range over a set number of periods. The indicator can range from 0 to 100. The closing price tends to close near the high in an uptrend and near the low in a downtrend.')

# Display Stochastic Oscillator chart
st.line_chart(data['momentum_stoch'])

# Determine signals based on Stochastic Oscillator
latest_stochastic = data['momentum_stoch'].iloc[-1]

if latest_stochastic < stochastic_buy_threshold:
    signal = "Buy"
elif latest_stochastic > stochastic_sell_threshold:
    signal = "Sell"
else:
    signal = "Neutral"

# Display the signal
st.write(f"Latest Stochastic Oscillator: {latest_stochastic}")
st.write(f"Signal: {signal}")




# Define CCI thresholds for buy and sell signals
cci_buy_threshold = -100
cci_sell_threshold = 100

# Create a Streamlit app
st.title("Commodity Channel Index (CCI) Signal App")
st.subheader('The Commodity Channel Index (CCI) is a technical indicator that measures the difference between the current price and the historical average price. When the CCI is above zero, it indicates the price is above the historic average. Conversely, when the CCI is below zero, the price is below the historic average.')

# Display CCI chart
st.line_chart(data['trend_cci'])

# Determine signals based on CCI
latest_cci = data['trend_cci'].iloc[-1]

if latest_cci < cci_buy_threshold:
    signal = "Buy"
elif latest_cci > cci_sell_threshold:
    signal = "Sell"
else:
    signal = "Neutral"

# Display the signal
st.write(f"Latest CCI: {latest_cci}")
st.write(f"Signal: {signal}")








# Define Williams %R thresholds for buy and sell signals
williams_r_buy_threshold = -80
williams_r_sell_threshold = -20

# Create a Streamlit app
st.title("Williams %R Signal App")

# Display Williams %R chart
st.line_chart(data['momentum_wr'])

# Determine signals based on Williams %R
latest_wr = data['momentum_wr'].iloc[-1]

if latest_wr < williams_r_buy_threshold:
    signal = "Buy"
elif latest_wr > williams_r_sell_threshold:
    signal = "Sell"
else:
    signal = "Neutral"

# Display the signal
st.write(f"Latest Williams %R: {latest_wr}")
st.write(f"Signal: {signal}")







# Define ROC thresholds for buy and sell signals
roc_buy_threshold = 0
roc_sell_threshold = 0

# Create a Streamlit app
st.title("Rate of Change (ROC) Signal App")

# Display ROC chart
st.line_chart(data['momentum_roc'])

# Determine signals based on ROC
latest_roc = data['momentum_roc'].iloc[-1]

if latest_roc > roc_buy_threshold:
    signal = "Buy"
elif latest_roc < roc_sell_threshold:
    signal = "Sell"
else:
    signal = "Neutral"

# Display the signal
st.write(f"Latest ROC: {latest_roc}")
st.write(f"Signal: {signal}")







data['stoch_oscillator'] = data['momentum_stoch_signal']

# Define Stochastic Oscillator thresholds for buy and sell signals
stoch_osc_buy_threshold = 20
stoch_osc_sell_threshold = 80

# Create a Streamlit app
st.title("Stochastic Oscillator (9,6) Signal App")

# Display the data
st.write(data.tail())

# Display Stochastic Oscillator (9,6) chart
st.line_chart(data['stoch_oscillator'])

# Determine signals based on Stochastic Oscillator (9,6)
latest_stoch_osc = data['stoch_oscillator'].iloc[-1]

if latest_stoch_osc < stoch_osc_buy_threshold:
    signal = "Buy"
elif latest_stoch_osc > stoch_osc_sell_threshold:
    signal = "Sell"
else:
    signal = "Neutral"

# Display the signal
st.write(f"Latest Stochastic Oscillator (9,6): {latest_stoch_osc}")
st.write(f"Signal: {signal}")





data['macd'] = data['trend_macd']

# Define MACD thresholds for buy and sell signals
macd_buy_threshold = 0
macd_sell_threshold = 0

# Create a Streamlit app
st.title("MACD (12,26) Signal App")

# Display the data
st.write(data.tail())

# Display MACD (12,26) chart
st.line_chart(data['macd'])

# Determine signals based on MACD (12,26)
latest_macd = data['macd'].iloc[-1]

if latest_macd > macd_buy_threshold:
    signal = "Buy"
elif latest_macd < macd_sell_threshold:
    signal = "Sell"
else:
    signal = "Neutral"

# Display the signal
st.write(f"Latest MACD (12,26): {latest_macd}")
st.write(f"Signal: {signal}")




data['adx'] = data['trend_adx']

# Define ADX thresholds for buy and sell signals
adx_buy_threshold = 20
adx_sell_threshold = 50

# Create a Streamlit app
st.title("Average Directional Index (ADX 14) Signal App")


# Display ADX (14) chart
st.line_chart(data['adx'])

# Determine signals based on ADX (14)
latest_adx = data['adx'].iloc[-1]

if latest_adx > adx_sell_threshold:
    signal = "Strong Trend"
elif latest_adx > adx_buy_threshold:
    signal = "Weak Trend"
else:
    signal = "No Clear Trend"

# Display the signal
st.write(f"Latest ADX (14): {latest_adx}")
st.write(f"Signal: {signal}")






# Calculate High-Low Index (14)
data['High_Low_Index'] = (data['High'] - data['Low']).rolling(window=14).sum()

# Create a Streamlit app
st.title("High-Low Index (14) Signal App")

# Display the data
st.write(data.tail())

# Display High-Low Index (14) chart
st.line_chart(data['High_Low_Index'])

# Determine signals based on High-Low Index (14)
latest_high_low_index = data['High_Low_Index'].iloc[-1]

# Define thresholds for buy and sell signals (adjust as needed)
buy_threshold = 500  # Example value, adjust as per your analysis
sell_threshold = 1000  # Example value, adjust as per your analysis

if latest_high_low_index < buy_threshold:
    signal = "Buy"
elif latest_high_low_index > sell_threshold:
    signal = "Sell"
else:
    signal = "Neutral"

# Display the signal
st.write(f"Latest High-Low Index (14): {latest_high_low_index}")
st.write(f"Signal: {signal}")






# Define the periods for short, medium, and long-term calculations
periods = [7, 14, 28]

# Calculate True Range
data['TrueRange'] = data['High'].combine(data['Close'].shift(1), max) - data['Low'].combine(data['Close'].shift(1), min)
data['TrueRange'] = data['TrueRange'].shift(1)

# Calculate Average True Range (ATR)
for period in periods:
    data[f'ATR_{period}'] = data['TrueRange'].rolling(window=period).mean()

# Calculate Buying Pressure (BP) and Selling Pressure (SP)
data['BP'] = data['Close'] - data['Low']
data['SP'] = data['High'] - data['Close']

# Calculate raw Ultimate Oscillator
for period in periods:
    data[f'Raw_Ult_Osc_{period}'] = (
        data['BP'].rolling(window=period).sum() / data[f'ATR_{period}'].rolling(window=period).sum()
        + data['BP'].shift(period)
        / data[f'ATR_{period}'].rolling(window=period).sum() * 2
        + data['SP'].rolling(window=period).sum() / data[f'ATR_{period}'].rolling(window=period).sum() * 4
    )

# Calculate Ultimate Oscillator
data['Ultimate_Oscillator'] = (
    data['Raw_Ult_Osc_7'] * 4
    + data['Raw_Ult_Osc_14'] * 2
    + data['Raw_Ult_Osc_28']
) / 7

# Create a Streamlit app
st.title("Ultimate Oscillator Signal App")

# Display the data
st.write(data.tail())

# Display Ultimate Oscillator chart
st.line_chart(data['Ultimate_Oscillator'])

# Determine signals based on Ultimate Oscillator
latest_ultimate_oscillator = data['Ultimate_Oscillator'].iloc[-1]

# Define thresholds for buy and sell signals (adjust as needed)
buy_threshold = 50  # Example value, adjust as per your analysis
sell_threshold = 70  # Example value, adjust as per your analysis

if latest_ultimate_oscillator < buy_threshold:
    signal = "Buy"
elif latest_ultimate_oscillator > sell_threshold:
    signal = "Sell"
else:
    signal = "Neutral"

# Display the signal
st.write(f"Latest Ultimate Oscillator: {latest_ultimate_oscillator}")
st.write(f"Signal: {signal}")





# Function to get stock data and generate forecast
def get_stock_forecast(ticker):
    two_years_ago = "2021-01-01"
    today = pd.to_datetime('today').strftime('%Y-%m-%d')
    
    # Download stock data
    data = yf.download(ticker, start=two_years_ago, end=today)
    data.insert(0, "Date", data.index, True)
    
    # Prepare data for Prophet
    df1 = data[['Date', 'Close']]
    df1 = df1.rename(columns={'Date': 'ds', 'Close': 'y'})
    
    # Create and fit the model
    model = Prophet()
    model.fit(df1)
    
    # Make future dataframe for forecasting
    future = model.make_future_dataframe(periods=365)
    
    # Generate forecast
    forecast = model.predict(future)
    
    return data, model, forecast

# Streamlit app
st.title(f'{ticker} Stock Price Forecast with Prophet')

# Get forecast data
data, model, forecast = get_stock_forecast(ticker)


# Display forecast plot
st.subheader('Stock Price Forecast')
fig1 = model.plot(forecast)
st.pyplot(fig1)
