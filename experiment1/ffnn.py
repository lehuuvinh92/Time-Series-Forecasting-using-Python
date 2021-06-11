# -*- coding: utf-8 -*-
"""
Created on Sun May  6 10:10:07 2018

@author: ACER
"""

from utils import Configuration
from utils import root_mean_squared_error
from utils import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers
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
#create data for ffnn
n_steps = 3
btc_x, btc_y = create_data_ann(close_btc, n_steps)
size = int(len(btc_x)*0.2)
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
#build ann
model = Sequential()
model.add(Dense(output_dim = 6, activation = 'relu', input_dim = n_steps))
model.add(Dropout(0.2))
model.add(Dense(output_dim = 4, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(output_dim = 1))
#sgd = optimizers.SGD(lr=0.01, decay = 0.0, momentum=0.9, nesterov=False)
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(x_train, y_train, batch_size = 1, nb_epoch = 500)
#Prediction and evaluating the model
y_pred = model.predict(x_test)
y_pred = sc_y.inverse_transform(y_pred)
y_pred = y_pred.reshape(y_pred.shape[0], )
#Error
rmse = root_mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
print('RMSE: %f, MAPE: %f' % (rmse, mape), '%')

# plot baseline and predictions
plt.figure(figsize=(16,12))
fig = plt.figure()
#    plt.plot(df_array[:, 3], label='Observed', color='#006699');
#    plt.plot(train_predict_plot, label='Prediction for Train Set', color='#006699', alpha=0.5);
plt.plot(y_test, label='Observed', color='#006699')
plt.plot(y_pred, label='Predicted', color='#ff0066')
plt.legend(loc='best')
plt.title('Expected Vs Predicted Views Forecasting - ANN')
plt.show()
fig.savefig('images/Hidden-2-Node100-Epoch-20.png', dpi=1000)

    