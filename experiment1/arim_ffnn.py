# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 22:27:17 2019

@author: ACER
"""

from utils import Configuration
from utils import root_mean_squared_error
from utils import mean_absolute_percentage_error
from utils import write_list_to_file
from utils import read_list_from_file
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from pandas import DataFrame
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler

#configs = Configuration()
#data = pd.read_csv(configs.dataset_path)
#date_btc = data['Date']
#close_btc = data['Close']
##Preprocessing bitcoin data
#ts_log = np.log(close_btc)
#ts_log_diff = ts_log - ts_log.shift()
#ts_log_diff.dropna(inplace=True)
##Prediction
#size = int(len(ts_log)*0.7)
## Divide into train and test
#train_arima, test_arima = ts_log[0:size], ts_log[size:len(ts_log)]
#history = [x for x in train_arima]
#predictions = list()
#originals = list()
#error_list = list()
#resid_arima = list()
#print('Printing Predicted vs Expected Values...')
#for t in test_arima:
#    model = ARIMA(history, order=(6, 1, 0))
#    model_fit = model.fit(disp=0)
#    output = model_fit.forecast()
#    pred_value = output[0]
#    original_value = t
#    history.append(original_value)
#    pred_value = np.exp(pred_value)
#    original_value = np.exp(original_value)
#    resid_arima.append(float(original_value - pred_value))
#    # Calculating the error
#    error = (abs(pred_value - original_value) / original_value) * 100
#    error_list.append(error)
#    print('predicted = %f,   expected = %f,   error = %f' % (pred_value, original_value, error), '%')
#    predictions.append(float(pred_value))
#    originals.append(float(original_value))

#Write originals, predictions and resid_arima
#write_list_to_file('originals.txt', originals)
#write_list_to_file('predictions.txt', predictions)
#write_list_to_file('resid_arima.txt', resid_arima)
#Read originals, predictions and resid_arima
originals = read_list_from_file('originals.txt')
predictions = read_list_from_file('predictions.txt')
resid_arima = read_list_from_file('resid_arima.txt')
originals = [float(x) for x in originals]
predictions = [float(x) for x in predictions]
resid_arima = [float(x) for x in resid_arima]
def create_data_ann(resid_arima, lag):
    btc_x = list()
    btc_y = list()
    for x in range(lag, len(resid_arima)):
        temp_x = list()
        for i in range(x-lag, x):
            temp_x.append(resid_arima[i])
        btc_x.append(temp_x)
        btc_y.append(resid_arima[x])
    return btc_x, btc_y

resid_arima = np.asarray(resid_arima)
resid_arima = resid_arima.ravel()
n_steps = 3
ann_size = int(len(resid_arima)*0.7)
btc_x, btc_y = create_data_ann(resid_arima, n_steps)
btc_x = np.asarray(btc_x)
btc_y = np.asarray(btc_y)
y_test = btc_y[len(btc_y)-ann_size:]
#scaling feature
sc_x = MinMaxScaler()
sc_y = MinMaxScaler()
btc_x = sc_x.fit_transform(btc_x)
btc_y = sc_y.fit_transform(btc_y.reshape(-1,1))
#split train, test dataset
x_train = btc_x[:len(btc_x)-ann_size]
x_test = btc_x[len(btc_x)-ann_size:]
y_train = btc_y[:len(btc_y)-ann_size]

#build ann
model = Sequential()
model.add(Dense(output_dim = 7, activation = 'relu', input_dim = n_steps))
model.add(Dropout(0.2))
model.add(Dense(output_dim = 1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(x_train, y_train, batch_size = 1, nb_epoch = 100)
#Prediction and evaluating the model
y_pred = model.predict(x_test)
y_pred = sc_y.inverse_transform(y_pred)
y_pred = y_pred.reshape(y_pred.shape[0], )
arima_pred = np.asarray(predictions[-ann_size:])
y_pred = y_pred.ravel()
arima_ann_pred = arima_pred + y_pred
# Calculating the error
for i in range(len(y_test)):
    error = (abs(arima_ann_pred[i] - originals[-ann_size:][i]) / originals[-ann_size:][i]) * 100
    print('predicted = %f,   expected = %f,   error = %f' % (arima_ann_pred[i], originals[-ann_size:][i], error), '%')

#Error
rmse = root_mean_squared_error(originals[-ann_size:], arima_ann_pred)
mape = mean_absolute_percentage_error(np.asarray(originals[-ann_size:]), np.asarray(arima_ann_pred))
print('RMSE: %f, MAPE: %f' % (rmse, mape), '%')

# plot baseline and predictions
plt.figure(figsize=(16,12))
fig = plt.figure()
plt.plot(originals[-ann_size:], label='Observed', color='#006699')
plt.plot(arima_ann_pred, label='Predicted', color='#ff0066')
plt.legend(loc='best')
plt.title('Expected Vs Predicted Views Forecasting - ANN')
plt.show()
fig.savefig('images/Hidden-2-Node100-Epoch-20.png', dpi=1000)