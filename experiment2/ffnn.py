# -*- coding: utf-8 -*-
"""
Created on Sun May  6 10:10:07 2018

@author: ACER
"""

import tsf_utils as tsfut
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model

def build_ffnn_model(train_set, n_lags, hidden_nodes=2, n_batch_size=1, epochs=10):
    """Build a ffnn model.
       train_set: train set after normalization
       n_lags: number of input
       epochs: one forward pass and one backward pass of 
       all the training examples
    """
    model = Sequential()
    model.add(Dense(output_dim=hidden_nodes, activation='relu', input_dim=n_lags))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim = 1))
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    x_train, y_train = tsfut.prepare_numpy_array_ann(train_set)
    model.fit(x_train, y_train, batch_size=n_batch_size, nb_epoch=epochs)
    return model

def make_prediction_ffnn(model, test_set, scaler):
     """Make predictions for test_set."""
     x_test, y_test = tsfut.prepare_numpy_array_ann(test_set)
     y_pred = model.predict(x_test)
     y_pred = scaler.inverse_transform(y_pred)
     y_pred = y_pred.reshape(y_pred.shape[0])
     
     y_test = y_test.reshape(-1, 1)
     y_test = scaler.inverse_transform(y_test)
     y_test = y_test.reshape(y_test.shape[0])
     return y_pred, y_test
 
def ffnn_model(ratio, n_lags, hidden_nodes, epochs):
    """Building and making prediction with ffnn model."""
    close_btc = tsfut.load_time_series_data()
    close_btc = close_btc.values
    close_btc = close_btc.reshape(-1, 1)
    close_btc, scaler = tsfut.scale_time_series(close_btc, ratio=ratio)
    train_set, test_set = tsfut.create_dataset_ann(close_btc, lags=n_lags, \
                                                   ratio=ratio)
    print("Running ffnn model...")
    ffnn_model = build_ffnn_model(train_set, n_lags, \
                                      hidden_nodes=hidden_nodes, epochs=epochs)
    model_name='models/ffnn/ratio_%f_nlags_%s_hidden_nodes_%s_epochs_%s_'\
                      % (ratio, n_lags, hidden_nodes, epochs) + 'ffnn_model.h5'
    ffnn_model.save(model_name)
    y_pred, y_test = make_prediction_ffnn(ffnn_model, test_set, scaler)
    rmse = tsfut.root_mean_squared_error(y_pred, y_test)
    mape = tsfut.mean_absolute_percentage_error(y_pred, y_test)
    return rmse, mape

def trained_ffnn_model(ratio, n_lags, model_path):
    """Load trained ffnn model and make prediction."""
    close_btc = tsfut.load_time_series_data()
    close_btc = close_btc.values
    close_btc = close_btc.reshape(-1, 1)
    close_btc, scaler = tsfut.scale_time_series(close_btc, ratio=ratio)
    train_set, test_set = tsfut.create_dataset_ann(close_btc, lags=n_lags, \
                                                   ratio=ratio)
    ffnn_model = load_model(model_path)
    y_pred, y_test = make_prediction_ffnn(ffnn_model, test_set, scaler)
    return y_pred, y_test