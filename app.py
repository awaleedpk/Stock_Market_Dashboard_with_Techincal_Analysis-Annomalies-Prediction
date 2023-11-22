import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import date, timedelta, datetime
import yfinance as yf
import streamlit as st
from plotly.subplots import make_subplots
from stocknews import StockNews
from prophet import Prophet
from hijri_converter import Hijri, Gregorian, convert
# ==============================================================================
# HOT FIX FOR YFINANCE .INFO METHOD
# Ref: https://github.com/ranaroussi/yfinance/issues/1729
# ==============================================================================

import requests
import urllib


class YFinance:
    user_agent_key = "User-Agent"
    user_agent_value = ("Mozilla/5.0 (Windows NT 6.1; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/58.0.3029.110 Safari/537.36")

    def __init__(self, ticker):
        self.yahoo_ticker = ticker

    def __str__(self):
        return self.yahoo_ticker

    def _get_yahoo_cookie(self):
        cookie = None

        headers = {self.user_agent_key: self.user_agent_value}
        response = requests.get("https://fc.yahoo.com",
                                headers=headers,
                                allow_redirects=True)

        if not response.cookies:
            raise Exception("Failed to obtain Yahoo auth cookie.")

        cookie = list(response.cookies)[0]

        return cookie

    def _get_yahoo_crumb(self, cookie):
        crumb = None

        headers = {self.user_agent_key: self.user_agent_value}

        crumb_response = requests.get(
            "https://query1.finance.yahoo.com/v1/test/getcrumb",
            headers=headers,
            cookies={cookie.name: cookie.value},
            allow_redirects=True,
        )
        crumb = crumb_response.text

        if crumb is None:
            raise Exception("Failed to retrieve Yahoo crumb.")

        return crumb

    @property
    def info(self):
        # Yahoo modules doc informations :
        # https://cryptocointracker.com/yahoo-finance/yahoo-finance-api
        cookie = self._get_yahoo_cookie()
        crumb = self._get_yahoo_crumb(cookie)
        info = {}
        ret = {}

        headers = {self.user_agent_key: self.user_agent_value}

        yahoo_modules = ("assetProfile,"  # longBusinessSummary
                         "summaryDetail,"
                         "financialData,"
                         "indexTrend,"
                         "defaultKeyStatistics")

        url = ("https://query1.finance.yahoo.com/v10/finance/"
               f"quoteSummary/{self.yahoo_ticker}"
               f"?modules={urllib.parse.quote_plus(yahoo_modules)}"
               f"&ssl=true&crumb={urllib.parse.quote_plus(crumb)}")

        info_response = requests.get(url,
                                     headers=headers,
                                     cookies={cookie.name: cookie.value},
                                     allow_redirects=True)

        info = info_response.json()
        info = info['quoteSummary']['result'][0]

        for mainKeys in info.keys():
            for key in info[mainKeys].keys():
                if isinstance(info[mainKeys][key], dict):
                    try:
                        ret[key] = info[mainKeys][key]['raw']
                    except (KeyError, TypeError):
                        pass
                else:
                    ret[key] = info[mainKeys][key]

        return ret

# ==============================================================================
# Header
# ==============================================================================

def render_sidebar():
    # Create two columns in the sidebar
    data_source_col1, data_source_col2 = st.sidebar.columns([1, 1])

    # Display the data source information
    data_source_col1.write("Data source:")
    data_source_col2.image('./img/yahoo_finance.png', width=100)
    global ticker_list
    # Get the list of stock tickers from S&P500
    
    
    
    ticker_list = pd.read_html(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
    global ticker

    # Add the ticker selection box
    ticker = st.sidebar.selectbox("Select Ticker", ticker_list)
    global start_date, end_date

    # Add the date input boxes for start and end dates
    start_date = st.sidebar.date_input(
        "Select Start Date", datetime.today().date() - timedelta(days=30))
    end_date = st.sidebar.date_input(
        "Select End Date", datetime.today().date())

    st.sidebar.subheader("")

    # Add an "Update" button and call the update_stock_data function when clicked
    if st.sidebar.button("Update"):
        if ticker:
            st.write(f"Updating stock data for {ticker}...")

    return ticker, start_date, end_date


# ==============================================================================
# Tab 1
# ==============================================================================

@st.cache_data
def GetCompanyInfo(ticker):
    return YFinance(ticker).info


@st.cache_data
def GetStockData(ticker, start_date, end_date):
    stock_df = yf.Ticker(ticker).history(start=start_date, end=end_date)
    stock_df.reset_index(inplace=True)
    stock_df['Date'] = stock_df['Date'].dt.date
    return stock_df


@st.cache_data
def GetStockData2(ticker, start_date, end_date, select_period):
    if select_period == 'Selected Range':
        stock_price = yf.Ticker(ticker).history(
            start=start_date, end=end_date, interval='1d').reset_index()
    else:
        stock_price = yf.Ticker(ticker).history(
            period=select_period, interval='1d').reset_index()

    return stock_price


def render_tab1():

    col4, col5, col6, col7 = st.columns((1, 1, 1, 1))
    company_data = GetCompanyInfo(ticker)

    # Fetch historical stock prices using yfinance
    stock_data = yf.Ticker(ticker).history(period='1y')

    # Extract relevant columns (Date and Close)
    stock_price = stock_data[['Close']].reset_index()
    # Inside the col4 block where you display the price metric
    yesterday_close_price = stock_price.iloc[-2]['Close'] if len(
        stock_price) >= 2 else None
    with col4:
        current_price = company_data.get("currentPrice", 'N/A')

        if yesterday_close_price is not None:
            delta_price = (
                (current_price - yesterday_close_price) / yesterday_close_price) * 100
            delta_price_formatted = f"({delta_price:.2f}%)"
        else:
            delta_price_formatted = 'N/A'

        st.metric(label="Price", value=current_price,
                  delta=delta_price_formatted)

    with col5:
        market_cap_value = company_data.get('marketCap', 'N/A')
        formatted_market_cap = "{:,.3f}B".format(
            market_cap_value / 1e9) if market_cap_value != 'N/A' else 'N/A'
        st.metric(label="Market Cap", value=formatted_market_cap)

    with col6:
        st.metric(label="Currency", value=company_data.get('currency'))

    with col7:
        period = st.selectbox(
            "Select Period", ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "MAX"], index=3)

    st.write("")
    st.write("")

    col1, col2, col3 = st.columns([1, 1, 2])

    if ticker != '':
        info = GetCompanyInfo(ticker)

        first_half_keys = {'previousClose': 'Previous Close',
                           'open': 'Open',
                           'bid': 'Bid',
                           'ask': 'Ask',
                           'fiftyTwoWeekLow': '52 Week Range',
                           'volume': 'Volume',
                           'averageVolume': 'Average Volume'}

        second_half_keys = {'beta': 'Beta',
                            'trailingPE': 'PE Ratio (TTM)',
                            'trailingEps': 'EPS (TTM)',
                            'earningsDate': 'Earnings Date',
                            'dividendRate': 'Forward Dividend',
                            'exDividendDate': 'Ex-Dividend Date',
                            'oneYearTargetPrice': '1y Target Est'}
    with col1:
        st.markdown("<br>" * 1, unsafe_allow_html=True)  # Create 2 empty lines
        for key in first_half_keys:
            if key == 'fiftyTwoWeekLow':
                st.write(
                    f"**{first_half_keys[key]}:** {info.get(key, 'N/A')} - {info.get('fiftyTwoWeekHigh', 'N/A')}")
            elif key in {'volume', 'averageVolume'}:
                st.write(
                    f"**{first_half_keys[key]}:** {info.get(key, 'N/A'):,.0f}")
            else:
                st.write(f"**{first_half_keys[key]}:** {info.get(key, 'N/A')}")

    with col2:
        st.markdown("<br>" * 1, unsafe_allow_html=True)  # Create 2 empty lines
        for key in second_half_keys:
            if key == 'exDividendDate':
                ex_dividend_date_timestamp = info.get(key)
                if ex_dividend_date_timestamp is not None:
                    ex_dividend_date = datetime.utcfromtimestamp(
                        ex_dividend_date_timestamp).strftime('%b %d, %Y')
                    st.write(
                        f"**{second_half_keys[key]}:** {ex_dividend_date}")
                else:
                    st.write(f"**{second_half_keys[key]}:** N/A")
            else:
                st.write(
                    f"**{second_half_keys[key]}:** {info.get(key, 'N/A')}")

    with col3:

        @st.cache_data
        def GetStockData(ticker, period):
            end_date = datetime.now()
            if period == "1M":
                start_date = end_date - timedelta(days=30)
            elif period == "3M":
                start_date = end_date - timedelta(days=90)
            elif period == "6M":
                start_date = end_date - timedelta(days=180)
            elif period == "YTD":
                start_date = datetime(end_date.year, 1, 1)
            elif period == "1Y":
                start_date = end_date - timedelta(days=365)
            elif period == "3Y":
                start_date = end_date - timedelta(days=3*365)
            elif period == "5Y":
                start_date = end_date - timedelta(days=5*365)
            stock_df = yf.Ticker(ticker).history(
                start=start_date, end=end_date, interval="1d")
            stock_df.reset_index(inplace=True)
            stock_df['Date'] = stock_df['Date'].dt.date
            return stock_df

        if ticker != '':
            stock_price = GetStockData(ticker, period)
            st.area_chart(data=stock_price, x='Date', y='Close',
                          color="#ADD8E6", use_container_width=True)


# ==============================================================================
# Tab 2
# ==============================================================================
@st.cache_data
def get_stock_data(ticker, start_date, end_date, selected_duration, selected_interval):
    if selected_duration == 'MAX':
        return yf.Ticker(ticker).history(period='max', interval=selected_interval)
    elif selected_duration == 'Selected Range':
        return yf.Ticker(ticker).history(start=start_date, end=end_date, interval='1d').reset_index()
    else:
        return yf.Ticker(ticker).history(period=selected_duration, interval=selected_interval)


def render_tab2():
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        selected_duration = st.selectbox("Select Period", [
                                         'Selected Range', '1M', '3M', '6M', 'YTD', '1Y', '3Y', '5Y', 'MAX'], index=0)

    with col2:
        interval_options = ['1d', '1mo', '1y']
        selected_interval = st.selectbox("Select Interval", interval_options)

    with col3:
        chart_type_options = ['Line', 'Candle']
        selected_chart_type = st.selectbox(
            "Select Chart Type", chart_type_options)

    show_sma = st.checkbox("Show Moving Average", value=True)
    show_volume = st.checkbox("Show Volume", value=True)

    stock_price = get_stock_data(
        ticker, start_date, end_date, selected_duration, selected_interval)

    M_Average = pd.DataFrame(yf.Ticker(ticker).history(
        period='max', interval=selected_interval)).reset_index()
    M_Average['M50'] = M_Average['Close'].rolling(50).mean()
    M_Average = M_Average[M_Average['Date'].isin(stock_price.index)]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if show_sma:
        fig.add_trace(go.Scatter(name='Moving Average/ 50 days',
                      x=M_Average['Date'], y=M_Average['M50'], marker_color='turquoise'), secondary_y=True)

    # Define the scaling factor for the maximum volume
    scaling_factor = 1.5
    
    # Check if volume should be displayed and if there is valid volume data
    if show_volume and not stock_price['Volume'].empty and not stock_price['Volume'].isnull().all():
        # Calculate an appropriate upper limit for the y-axis using the modified scaling factor
        max_volume = max(stock_price['Volume']) * scaling_factor
        
        # Add a bar trace for the volume of shares to the Plotly figure
        fig.add_trace(go.Bar(
            name='Volume of Shares',
            x=stock_price.index,
            y=stock_price['Volume']
        ))
        
        # Update the layout to set the y-axis range for the volume plot
        fig.update_layout(
            yaxis1=dict(range=[0, max_volume]),
            title='Stock Price with Volume',
            xaxis_title='Date',
            yaxis_title='Volume'
        )

    if selected_chart_type == 'Line':
        fig.add_trace(go.Scatter(name='Close Value', x=stock_price.index,
                      y=stock_price['Close']), secondary_y=True)

    elif selected_chart_type == 'Candle':
        fig.add_trace(go.Candlestick(name='Stock value (Open, High, Low, Close)', x=stock_price.index,
                                     open=stock_price['Open'],
                                     high=stock_price['High'],
                                     low=stock_price['Low'],
                                     close=stock_price['Close']), secondary_y=True)

    fig.update_layout(xaxis_rangeslider_visible=True, autosize=True)
    st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# Tab 3
# ==============================================================================


def render_tab3():
    # Create a Streamlit form for financial statement selection
    financial_form = st.form(key='financial')

    # Dropdown for selecting the type of financial info
    selected_statement_type = financial_form.selectbox("Select financial statement type", [
                                                       'Income Statement', 'Balance Sheet', 'Cash Flow'])

    # Radio buttons for selecting the time frame (Annual or Quarterly)
    selected_time_frame = financial_form.radio(
        '', ['Annual', 'Quarterly'], horizontal=True)

    # Form submission button
    financial_form.form_submit_button('Show statements')

    # Function to retrieve company financials based on user selections
    def get_company_financials(ticker, statement_type, time_frame):
        financial_func_map = {
            'Balance Sheet': yf.Ticker(ticker).balance_sheet if time_frame == 'Annual' else yf.Ticker(ticker).quarterly_balance_sheet,
            'Income Statement': yf.Ticker(ticker).financials if time_frame == 'Annual' else yf.Ticker(ticker).quarterly_financials,
            'Cash Flow': yf.Ticker(ticker).cashflow if time_frame == 'Annual' else yf.Ticker(ticker).quarterly_cashflow
        }
        return pd.DataFrame(financial_func_map.get(statement_type, {}))

    # Get financial data for the selected options, fill NaN values with 0, and format the date columns
    financial_data = get_company_financials(
        ticker, selected_statement_type, selected_time_frame).fillna(0)
    financial_data.columns = pd.to_datetime(financial_data.columns).strftime(
        '%m/%d/%Y')  # Convert to datetime and format

    # Calculate the Trailing Twelve Months (TTM) column for Annual reports
    if selected_time_frame == 'Annual' and 'TTM' not in financial_data.columns:
        financial_data['TTM'] = financial_data.iloc[:, -4:].sum(axis=1)

    financial_data = financial_data.style.format("${:0,.2f}", na_rep="N/A")

    # Apply custom CSS class for highlighting
    financial_data = financial_data.applymap(
        lambda x: 'background-color: #ffcccc' if pd.isnull(x) else '')

    # Display the header and financial data in a table
    st.header(selected_time_frame + ' ' + selected_statement_type)
    st.table(financial_data)


# ==============================================================================
# Tab 4
# ==============================================================================



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

def stock_price_forecast_app(ticker):
    # Streamlit app
    st.title(f'{ticker} Stock Price Forecast with Prophet')

    # Get forecast data
    data, model, forecast = get_stock_forecast(ticker)

    # Display forecast plot
    st.subheader('Stock Price Forecast')
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

def render_tab4():
    ticker_symbol = ticker  # Replace with the desired stock ticker symbol

    # Call functions within render_tab4
    data, model, forecast = get_stock_forecast(ticker_symbol)
    stock_price_forecast_app(ticker_symbol)


# ==============================================================================
# Tab 5
# ==============================================================================

def render_tab5():
    st.header(f'News for {ticker}')

    stock = [ticker]
    # Create a StockNews object for the selected stock
    sn = StockNews(stock, save_news=False)

    # Read the RSS feed for the selected stock and get the news data as a DataFrame
    df_news = sn.read_rss()
    # Check if there are articles available
    if df_news.empty:
        st.write("No news articles available.")
        return

    # Initialize variables to keep track of sentiments
    total_sentiment = 0

    # Determine the number of articles to iterate over
    num_articles = min(10, len(df_news))

    # Accumulate sentiments for each article
    for i in range(num_articles):
        title_sentiment = df_news['sentiment_title'][i]
        news_sentiment = df_news['sentiment_summary'][i]
        total_sentiment += (title_sentiment + news_sentiment)

    # Calculate the average sentiment only if there are articles
    if num_articles > 0:
        # Divide by 2 for title and news
        avg_sentiment = total_sentiment / (num_articles * 2)
        # Display the overall sentiment with a symbol (green if positive, red if negative)
        sentiment_display = '😌' if avg_sentiment >= 0.5 else '😥'

        # Add a border to the overall sentiment display
        st.markdown(
            f'<div style="border: 2px solid #e4e4e4; padding: 10px; border-radius: 5px; margin-bottom: 20px;">'
            f'<h3 style="margin: 0; padding: 0;">Overall Sentiment: {avg_sentiment:.2%} {sentiment_display}</h3>'
            f'</div>',
            unsafe_allow_html=True
        )

    # Iterate over the articles and display certain attributes
    for i in range(num_articles):
        st.subheader(f"**{df_news['title'][i]}**")
        published_date = df_news['published'][i]
        parsed_date = datetime.strptime(
            published_date, "%a, %d %b %Y %H:%M:%S %z")
        formatted_date = parsed_date.strftime("%m-%d-%Y")
        st.caption(f"Date Published: {formatted_date}")
        st.write("    ", df_news['summary'][i])

    else:
        st.write("No more news articles available.")

# ==============================================================================
# Tab 6
# ==============================================================================
from pytrends.request import TrendReq
import streamlit as st

def get_google_trends_data(ticker):
    # Set up pytrends
    pytrends = TrendReq(hl='en-US', tz=360)

    # Define the keyword (ticker) you want to search for
    kw_list = [ticker]

    # Build the payload for the API request
    pytrends.build_payload(kw_list, timeframe='today 1-m', geo='US')

    # Get the interest over time data
    interest_over_time_df = pytrends.interest_over_time()

    return interest_over_time_df

def plot_google_trends_data(ticker, interest_over_time_df):
    # Set up Streamlit
    st.title("Google Trends Ticker Plotter")

    # Plot the Google Trends data
    st.header(f'Google Trends for {ticker}')
    st.line_chart(interest_over_time_df[[ticker]])

# Example usage in Streamlit app
def render_tab6():
    ticker_symbol = ticker  # Replace with the desired stock ticker symbol

    # Call the function to get Google Trends data
    interest_over_time_df = get_google_trends_data(ticker_symbol)

    # Call the function to plot Google Trends data in Streamlit
    plot_google_trends_data(ticker_symbol, interest_over_time_df)



def download_stock_data(ticker):
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

    # Download data
    data = yf.download(ticker, start=two_years_ago, end=end_date)

    data.insert(0, "Date", data.index, True)

    return data

def calculate_weekday_effect(data):
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

    return current_day, weekday_avg

def display_weekday_effect(current_day, weekday_avg):
    st.header('1. Weekday Effect')
    st.text('Weekday Effect: Stock market trend influenced by day of the week, e.g., higher returns on Fridays.')
    
    # Print the current day
    st.write(f"Current Day: {current_day}")

    # Check if the percentage return for the current day is positive
    if weekday_avg[current_day] > 0:
        st.write('Signal: Buy')
    else:
        st.write('Signal: Negative')

def calculate_monthly_return_effect(data):
    # Convert datetime index to months
    data['Month'] = data.index.month

    # Calculate monthly returns
    data['Monthly_Returns'] = data['Close'].pct_change()

    # Group by month and calculate mean returns
    monthly_returns = data.groupby('Month')['Monthly_Returns'].mean()

    # Get current month number
    now = datetime.now()
    current_month = now.month

    return current_month, monthly_returns

def display_monthly_return_effect(current_month, monthly_returns):
    st.header("2. Monthly Return Effect")

    # Display current month
    st.write(f"Current Month: {current_month}")

    # Check if the percentage return for the current month is positive
    if monthly_returns[current_month] > 0:
        st.write('Signal: Buy')
    else:
        st.write('Signal: Sell')

def calculate_quarterly_effect(data):
    # Calculate the quarter for each data point
    data['Quarter'] = data.index.to_period('Q')

    # Calculate quarterly averages
    quarter_avg = data.groupby(data['Quarter'])['Percentage_Return'].mean()

    # Get the current quarter
    current_quarter = pd.to_datetime(datetime.today()).to_period('Q')

    return current_quarter, quarter_avg

def display_quarterly_effect(current_quarter, quarter_avg):
    st.header('3. Quarterly Effect')
    st.text('Quarterly Effect: Stock market trend influenced by Q1, Q2, Q3, Q4.')

    # Print the current quarter
    st.write(f"Current quarter: {current_quarter}")

    # Check if the average return for the current quarter is positive
    if quarter_avg.loc[current_quarter] > 0:
        st.write('Signal: Buy')
    else:
        st.write('Signal: Negative')


def calculate_day_of_month_effect(data):
    # Calculate the day of the month for each data point
    data['Day'] = data.index.day

    # Calculate daily averages
    day_avg = data.groupby(data['Day'])['Percentage_Return'].mean()

    # Get the current day of the month
    current_day = datetime.today().day

    return current_day, day_avg

def display_day_of_month_effect(current_day, day_avg):
    st.header('4. Day of the Month Effect')
    st.text('Day of the Month Effect: Stock market trend influenced by specific days of the month.')

    # Print the current day of the month
    st.write(f"Current day of the month: {current_day}")

    # Check if the average return for the current day of the month is positive
    if day_avg.loc[current_day] > 0:
        st.write('Signal: Buy')
    else:
        st.write('Signal: Negative')

def calculate_week_number_returns_effect(data):
    # Calculate the week number for each data point
    data['Week_Number'] = data.index.isocalendar().week
    # Calculate weekly averages
    week_avg = data.groupby('Week_Number')['Percentage_Return'].mean()
    # Get the current week number
    current_week = datetime.today().isocalendar()[1]
    return current_week, week_avg

def display_week_number_returns_effect(current_week, week_avg):
    st.header('5. Week Number Returns Effect')
    st.text('Week Number Returns Effect: Stock market trend influenced by the week number within a year.')

    # Print the current week number
    st.write(f"Current week number: {current_week}")

    # Check if the average return for the current week number is positive
    if week_avg.loc[current_week] > 0:
        st.write('Signal: Buy')
    else:
        st.write('Signal: Negative')


def calculate_islamic_month_returns_effect(data):
    # Function to convert Gregorian date to Islamic month
    def gregorian_to_islamic(date):
        hijri_date = convert.Gregorian(date.year, date.month, date.day).to_hijri()
        return hijri_date.month_name('en')

    # Convert dates to Islamic months
    data['Hijri_Months_EN'] = data.index.map(gregorian_to_islamic)

    # Calculate Islamic month returns
    Hijri_Months_Returns = data.groupby('Hijri_Months_EN')['Percentage_Return'].mean()

    # Get current Islamic month
    now = pd.to_datetime(date.today())
    current_month = gregorian_to_islamic(now)

    return current_month, Hijri_Months_Returns

def display_islamic_month_returns_effect(current_month, Hijri_Months_Returns):
    st.header("6. Islamic Month Returns Effect")

    # Display Islamic month returns
    st.write(Hijri_Months_Returns)

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







# Example usage in Streamlit app
def render_tab7():
    # Replace 'AAPL' with the desired stock ticker symbol
    ticker_symbol = ticker
    
    # Download data
    data = download_stock_data(ticker_symbol)

    # Render Weekday Effect
    current_day, weekday_avg = calculate_weekday_effect(data)
    display_weekday_effect(current_day, weekday_avg)

    # Render Monthly Return Effect
    current_month, monthly_returns = calculate_monthly_return_effect(data)
    display_monthly_return_effect(current_month, monthly_returns)

    # Render Quarterly Effect
    current_quarter, quarter_avg = calculate_quarterly_effect(data)
    display_quarterly_effect(current_quarter, quarter_avg)

    # Render Day of the Month Effect
    current_day, day_avg = calculate_day_of_month_effect(data)
    display_day_of_month_effect(current_day, day_avg)

    # Render Week Number Returns Effect
    current_week, week_avg = calculate_week_number_returns_effect(data)
    display_week_number_returns_effect(current_week, week_avg)


    # Render Islamic Month Returns Effect
    current_month_islamic, islamic_month_returns = calculate_islamic_month_returns_effect(data)
    display_islamic_month_returns_effect(current_month_islamic, islamic_month_returns)



# ==============================================================================
# Main body
# ==============================================================================
st.set_page_config(layout="wide", page_title="My Streamlit App",
                   page_icon=":chart_with_upwards_trend:")
st.title("Stock Research and Analysis")
# Render the header
render_sidebar()

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Company Profile", "Chart", "Financial Statements", "Forecast", "News (Sentiment Analysis)", "Google Trend", "Calendar Anomalies"])
with tab1:
    render_tab1()
with tab2:
    render_tab2()
with tab3:
    render_tab3()
with tab4:
    render_tab4()
with tab5:
    render_tab5()
with tab6:
    render_tab6()
with tab7:
    render_tab7()
###############################################################################
# END
###############################################################################
