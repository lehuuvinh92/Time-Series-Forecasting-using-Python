# -*- coding: utf-8 -*-
"""
Created on Sun May  6 10:09:25 2018
@author: ACER

"""
from utils import Configuration
from utils import root_mean_squared_error
from utils import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA

#Load and draw chart of dataset
configs = Configuration()
data = pd.read_csv(configs.dataset_path)
date_btc = data['Date']
datetime_btc = [dt.datetime.strptime(date, '%d-%b-%y').date() for date in date_btc]
close_btc = data['Close']
plt.plot(datetime_btc, close_btc)
plt.xlabel('Date')
plt.ylabel('Price in USD')
plt.title("Closing price distribution of bitcoin")
plt.show()
#Testing the Stationarity with Augmented Dicky Fuller Test
def test_stationarity(data):
    #Determing rolling statistics
    rolmean = data.rolling(window=22,center=False).mean()
    rolstd = data.rolling(window=12,center=False).std()
    #Plot rolling statistics:
    plt.plot(data, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey Fuller test    
    result=adfuller(data)
    print('ADF Stastistic: %f'%result[0])
    print('p-value: %f'%result[1])
    for key,value in result[4].items():
         if result[0]>value:
            print("The graph is non stationary")
            break
         else:
            print("The graph is stationary")
            break;
    print('Critical values:')
    for key,value in result[4].items():
        print('\t%s: %.3f ' % (key, value))    
#Identify p,d,q from training dataset
train_size = int(len(close_btc)*0.8)
train_db = close_btc[0:train_size]
test_stationarity(train_db)
train_db_log=np.log(train_db)
test_stationarity(train_db_log)
train_db_log_diff = train_db_log - train_db_log.shift()
train_db_log_diff.dropna(inplace=True)
test_stationarity(train_db_log_diff)
#Identifying ARIMA model
plot_acf(train_db_log_diff, lags=32)
plot_pacf(train_db_log_diff, lags=32)
#Estimating ARIMA model and check suitability of ARIMA model
model = ARIMA(train_db_log, order=(6,1,0)) 
results_ARIMA = model.fit(disp=-1)
print(results_ARIMA.summary())
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS (Residual sum of squares): %.7f'% sum((results_ARIMA.fittedvalues-train_db_log_diff)**2))
plt.show()
#Prediction
ts_log = np.log(close_btc)
plt.plot(ts_log,color="green")
plt.title('Closing Price with Log of Bitcoin')
plt.show()
# Divide into train and test
size = int(len(ts_log)*0.8)
train_arima, test_arima = ts_log[0:size], ts_log[size:len(ts_log)]
history = [x for x in train_arima]
predictions = list()
originals = list()
error_list = list()
print('Printing Predicted vs Expected Values...')
for t in test_arima:
    model = ARIMA(history, order=(6, 1, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    pred_value = output[0]
    original_value = t
    history.append(original_value)
    pred_value = np.exp(pred_value)
    original_value = np.exp(original_value)
    # Calculating the error
    error = (abs(pred_value - original_value) / original_value) * 100
    error_list.append(error)
    print('predicted = %f,   expected = %f,   error = %f' % (pred_value, original_value, error), '%')
    predictions.append(float(pred_value))
    originals.append(float(original_value))

rmse = root_mean_squared_error(originals, predictions)
mape = mean_absolute_percentage_error(np.asarray(originals), np.asarray(predictions))
print('RMSE: %f, MAPE: %f' % (rmse, mape), '%')

plt.figure(figsize=(16, 12))
plt.plot(predictions, label='Predicted', color= 'green')
plt.plot(originals, label='Observed', color = 'orange')
plt.legend(loc='best')
plt.title('Expected Vs Predicted Views Forecasting - ARIMA')
plt.show()