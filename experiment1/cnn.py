# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 20:17:55 2019

@author: ACER
"""

from utils import Configuration
from utils import root_mean_squared_error
from utils import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from sklearn.preprocessing import MinMaxScaler

configs = Configuration()
#read dataset
data = pd.read_csv(configs.dataset_path)
close_btc = data['Close']
close_btc = close_btc.values

def create_data_ann(close_btc, lag):
    btc_x = list()
    btc_y = list()
    for x in range(lag, len(close_btc)):
        temp_x = list()
        for i in range(x-lag, x):
            temp_x.append(close_btc[i])
        btc_x.append(temp_x)
        btc_y.append(close_btc[x])
    return btc_x, btc_y
#create data for cnn
n_steps = 7
btc_x, btc_y = create_data_ann(close_btc, n_steps)
size = int(len(btc_x)*0.05)
btc_x = np.asarray(btc_x)
btc_y = np.asarray(btc_y)
y_test = btc_y[len(btc_y)-size:]
#scaling feature
sc_x = MinMaxScaler()
sc_y = MinMaxScaler()
btc_x = sc_x.fit_transform(btc_x)
btc_y = sc_y.fit_transform(btc_y.reshape(-1,1))
#split train, test dataset
x_train = btc_x[:len(btc_x)-size]
x_test = btc_x[len(btc_x)-size:]
y_train = btc_y[:len(btc_y)-size]
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], n_features))
#define model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(70, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(x_train, y_train, epochs=200)
#Prediction and evaluating the model
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], n_features))
y_pred = model.predict(x_test, verbose=0)
y_pred = sc_y.inverse_transform(y_pred)
y_pred = y_pred.reshape(y_pred.shape[0], )
# Calculating the error
for i in range(len(y_test)):
    error = (abs(y_pred[i] - y_test[i]) / y_test[i]) * 100
    print('predicted = %f,   expected = %f,   error = %f' % (y_pred[i], y_test[i], error), '%')
#Error
rmse = root_mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
print('RMSE: %f, MAPE: %f' % (rmse, mape), '%')

# plot baseline and predictions
plt.figure(figsize=(16,12))
fig = plt.figure()
plt.plot(y_test, label='Observed', color='#006699')
plt.plot(y_pred, label='Predicted', color='#ff0066')
plt.legend(loc='best')
plt.title('Expected Vs Predicted Views Forecasting - ANN')
plt.show()
fig.savefig('images/Hidden-2-Node100-Epoch-20.png', dpi=1000)
