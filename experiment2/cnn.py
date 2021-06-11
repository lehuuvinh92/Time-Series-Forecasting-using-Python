# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 20:17:55 2019

@author: ACER
"""
import tsf_utils as tsfut
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from keras.layers import Dropout
from keras.models import load_model

def build_cnn_model(train_set, n_lags, n_features=1, hidden_nodes=2, \
                    n_batch_size=1, epochs=10):
    """Build a cnn model.
       train_set: train set after normalization
       n_lags: number of input
       epochs: one forward pass and one backward pass of 
       all the training examples
    """
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', \
                     input_shape=(n_lags, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(output_dim=hidden_nodes, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=1))
    model.compile(optimizer='adam', loss='mse')
    x_train, y_train = tsfut.prepare_numpy_array_ann(train_set)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], n_features))
    model.fit(x_train, y_train, batch_size = n_batch_size, nb_epoch=epochs)
    return model

def make_prediction_cnn(model, test_set, scaler, n_features=1):
     """Make predictions for test_set."""
     x_test, y_test = tsfut.prepare_numpy_array_ann(test_set)
     x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], n_features))
     y_pred = model.predict(x_test, verbose=0)
     y_pred = scaler.inverse_transform(y_pred)
     y_pred = y_pred.reshape(y_pred.shape[0])
     
     y_test = y_test.reshape(-1, 1)
     y_test = scaler.inverse_transform(y_test)
     y_test = y_test.reshape(y_test.shape[0])
     return y_pred, y_test
 
def cnn_model(ratio, n_lags, hidden_nodes, epochs):
    """Building and making prediction with cnn model."""
    close_btc = tsfut.load_time_series_data()
    close_btc = close_btc.values
    close_btc = close_btc.reshape(-1, 1)
    close_btc, scaler = tsfut.scale_time_series(close_btc, ratio=ratio)
    train_set, test_set = tsfut.create_dataset_ann(close_btc, lags=n_lags, \
                                                   ratio=ratio)
    print("Running cnn model...")
    cnn_model = build_cnn_model(train_set, n_lags, \
                                      hidden_nodes=hidden_nodes, epochs=epochs)
    model_name='models/cnn/ratio_%f_nlags_%s_hidden_nodes_%s_epochs_%s_'\
                       % (ratio, n_lags, hidden_nodes, epochs) + 'cnn_model.h5'
    cnn_model.save(model_name)
    y_pred, y_test = make_prediction_cnn(cnn_model, test_set, scaler)
    rmse = tsfut.root_mean_squared_error(y_pred, y_test)
    mape = tsfut.mean_absolute_percentage_error(y_pred, y_test)
    return rmse, mape

def trained_cnn_model(ratio, n_lags, model_path):
    """Load trained cnn model and make prediction."""
    close_btc = tsfut.load_time_series_data()
    close_btc = close_btc.values
    close_btc = close_btc.reshape(-1, 1)
    close_btc, scaler = tsfut.scale_time_series(close_btc, ratio=ratio)
    train_set, test_set = tsfut.create_dataset_ann(close_btc, lags=n_lags, \
                                                   ratio=ratio)
    cnn_model = load_model(model_path)
    y_pred, y_test = make_prediction_cnn(cnn_model, test_set, scaler)
    return y_pred, y_test