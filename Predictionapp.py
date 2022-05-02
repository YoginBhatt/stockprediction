import streamlit as st
from datetime import date
import matplotlib
import matplotlib.pyplot as plt
import pandas_datareader as data

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go



START = "2018-01-01"
TODAY = date.today().strftime("%Y-%m-%d")#select the current date in yyyy-mm-dd format


st.set_page_config(page_title="Stock Prediction Webapp")
st.title("Stock Prediction App")



stocks = ('CIPLA.NS',
'SHREECEM.NS',
'ULTRACEMCO.NS',
'HEROMOTOCO.NS',
'ICICIBANK.NS',
'NESTLEIND.NS',
'ITC.NS',
'BAJAJFINSV.NS',
'TATASTEEL.NS',
'BAJFINANCE.NS',
'TECHM.NS',
'INDUSINDBK.NS',
'TCS.NS',
'RELIANCE.NS',
'BHARTIARTL.NS',
'HINDALCO.NS',
'LT.NS',
'KOTAKBANK.NS',
'TATACONSUM.NS',
'APOLLOHOSP.NS',
'HDFCLIFE.NS',
'MARUTI.NS',
'TITAN.NS',
'NTPC.NS',
'BRITANNIA.NS',
'WIPRO.NS',
'ONGC.NS',
'BAJAJ-AUTO.NS',
'COALINDIA.NS',
'MM.NS',


'TSLA',
'AAPL',
'AZN',
'GBP/USD')
#'TCS.NS', 'LT.NS', 'INFY.NS', 'ITC.NS', 'MARUTI.NS', 'KOTAKBANK.NS', 'WIPRO.NS','TSLA', 'AAPL',
selected_stock = st.selectbox('Select stock symbol for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 5)# years for which we need to predict
period = n_years * 365

# loading the data from the yahoo finance
@st.cache # this will cache the data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data...  done!')# reseting the text

st.subheader('Raw data')
st.write(data.head())
st.write(data.tail())# pandas data frame
# printed the current data of the stock

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))# openig value to th stock
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))# closing value of the stock
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)# slider for graph
    st.plotly_chart(fig)


plot_raw_data()

# Predict forecast with Prophet and fbprophet
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"}) #rename the date and close beacuse fbprophet expect a certain format


#fbprophet model
m = Prophet()
m.fit(df_train)# fit the training data and start training
future = m.make_future_dataframe(periods=period)# for preditction we need data freame which goes to future
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.head(2500))
st.write(forecast.tail())# printing forcasting data

#ploting the forcasting data with different component
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)