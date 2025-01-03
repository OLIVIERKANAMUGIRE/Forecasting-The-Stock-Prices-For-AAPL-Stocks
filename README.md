# Forecasting The Stock Prices For AAPL Stocks
 
This project aims to forcast the stick price in a sequencial dataset.

Weekly submission will be made to indicate the progress if the project.

-Find the weekly updates on codes in the Week_Sub_Files Update



**Introduction**

The task of forecasting stock prices for Apple Inc. (AAPL) using historical data from 2006 to 2018 involves applying statistical and machine learning techniques to predict future price movements based on past trends. This analysis seeks to identify patterns, correlations, and trends within the stock's historical price data, including factors such as daily closing prices, trading volume, and market behavior. By leveraging advanced forecasting models—such as time series analysis, regression models, and neural networks—the goal is to develop a reliable predictive framework that can inform investment strategies, risk management, and decision-making in the stock market. In this study, we employ three machine learning models Recurrent Neural Networks (RNNs), Long Short-Term Memory networks (LSTMs), and the Autoformer model as they represent key advancements in time-series forecasting and sequential data modeling. We explain a bit of these methods in the literature review section [2]


# Data Description
The dataset consists of daily stock price data for Apple Inc. (AAPL) over 12 years, from January 1, 2006, to January 1, 2018. The data includes attributes related to Apple's stock performance, recorded for each trading day. The total number of records is 3,019 and the attributes are; Date, Open, High, Low, Close, Volume, and Name describing the trading date, price of the stock at market open, highest price of the stock during the trading day, lowest price of the stock during the trading day, stock's closing price at the end of the trading day, number of shares traded during the day and the ticker symbol (AAPL) of the stock respectively.

Find the data here <a href="https://www.marketwatch.com/investing/stock/aapl/downloaddatapartial?startdate=11/20/2024%2000:00:00&enddate=12/20/2024%2023:59:59&daterange=d30&frequency=p1d&csvdownload=true&downloadpartial=false&newdates=false">LINK</a> 

![Alt text](i3.png)

# Experiment and Results
Find more details here <a href = "Report.pdf"> Read More</a>

**SOME RESULTS**

The following are some results obtained from the implemented models

**Baseline Models**

AR Prediction 

<img src="ar model.png" width="500" />

**RNN Results**

**LSTM Result**

<img src="multivariatepredictions.png" width="500" />

**AUtoformer Result**

<img src="autoformer pred.png" width="500" />


# Contributors 

Bright Wiredu Nuakoh  <br />
Olivier Kanamugire <br />
Durbar Hore Partha
# Kiitos!