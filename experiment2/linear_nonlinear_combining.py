# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 19:44:17 2019

@author: ACER
"""
import tsf_utils as tsfut
import arima as ari
import ffnn
import cnn
import lstm
import svr
from itertools import product
import numpy as np
import pandas as pd
import os
from sklearn.externals import joblib

def arima_for_combining(ratio=0.8):
    """Building and making prediction with arima model."""
    close_btc = tsfut.load_time_series_data()
    train_set, test_set = tsfut.split_time_series_dataset(close_btc,\
                                            ratio=ratio)
    ari.test_stationarity(train_set)
    train_set_log=np.log(train_set)
    test_set_log=np.log(test_set)
    ari.test_stationarity(train_set_log)
    train_set_log_diff = train_set_log - train_set_log.shift()
    train_set_log_diff.dropna(inplace=True)
    ari.test_stationarity(train_set_log_diff)
    p, q = ari.identify_parameters_arima(train_set_log_diff)
    p = p[:3]
    p.insert(0, int(0))
    q = q[:3]
    q.insert(0, int(0))
    d = [1]
    parameters = product(p, d, q)
    parameters_list = list(parameters)
    arima_params = ari.select_arima_model(train_set_log, parameters_list)
    sorted_aic_params = pd.DataFrame(arima_params)
    sorted_aic_params.columns = ['parameters', 'aic']
    sorted_aic_params = sorted_aic_params.sort_values(by = 'aic', ascending=True)
    predicted_values = list()
    observed_values = list()
    print("Running arima model...")
    for index, row in sorted_aic_params.iterrows():
        param = row['parameters']
        try:
            predicted_values, observed_values = ari.make_prediction_arima(\
                                        train_set_log, test_set_log, param)
        except:
            print('Inappropriate model:', param)
            continue
        if predicted_values:
                print("Appropriate parameters:")
                print(param)
                break
    for i in range(len(predicted_values)):
        result = '%f,%f,%f' % (predicted_values[i], observed_values[i], \
                            float(observed_values[i] - predicted_values[i]))
        tsfut.write_string_to_file('results/cb_arima_result.txt', result)
        
def ffnn_for_combining(ratio, resid_set, n_lags, hidden_nodes, epochs):
    """Building and making prediction with ffnn model."""
    resid_set = pd.Series(resid_set)
    resid_set = resid_set.values
    resid_set = resid_set.reshape(-1, 1)
    resid_set, scaler = tsfut.scale_time_series(resid_set, ratio=ratio)
    train_set, test_set = tsfut.create_dataset_ann(resid_set, lags=n_lags, \
                                                   ratio=ratio)
    print("Running ffnn model for combining...")
    ffnn_model = ffnn.build_ffnn_model(train_set, n_lags, \
                                      hidden_nodes=hidden_nodes, epochs=epochs)
    model_name='models/nlags_%s_hidden_nodes_%s_epochs_%s_' % (n_lags, hidden_nodes,\
                                                     epochs) + 'cb_ffnn_model.h5'
    ffnn_model.save(model_name)
    y_pred, y_test = ffnn.make_prediction_ffnn(ffnn_model, test_set, scaler)
    return y_pred, y_test

def cnn_for_combining(ratio, resid_set, n_lags, hidden_nodes, epochs):
    """Building and making prediction with cnn model."""
    resid_set = pd.Series(resid_set)
    resid_set = resid_set.values
    resid_set = resid_set.reshape(-1, 1)
    resid_set, scaler = tsfut.scale_time_series(resid_set, ratio=ratio)
    train_set, test_set = tsfut.create_dataset_ann(resid_set, lags=n_lags, \
                                                   ratio=ratio)
    print("Running cnn model for combining...")
    cnn_model = cnn.build_cnn_model(train_set, n_lags, \
                                      hidden_nodes=hidden_nodes, epochs=epochs)
    model_name='models/nlags_%s_hidden_nodes_%s_epochs_%s_' % (n_lags, hidden_nodes,\
                                                     epochs) + 'cb_cnn_model.h5'
    cnn_model.save(model_name)
    y_pred, y_test = cnn.make_prediction_cnn(cnn_model, test_set, scaler)
    return y_pred, y_test

def svr_for_combining(ratio, resid_set, n_lags):
    """Building and making prediction with svr model."""
    resid_set = pd.Series(resid_set)
    resid_set = resid_set.values
    resid_set = resid_set.reshape(-1, 1)
    resid_set, scaler = tsfut.scale_time_series(resid_set, ratio=ratio)
    train_set, test_set = tsfut.create_dataset_ann(resid_set, lags=n_lags, \
                                                   ratio=ratio)
    print("Running svr model for combining...")
    svr_model = svr.build_svr_model(train_set, 'rbf')
    model_name='models/nlags_%s' % (n_lags) + '_cb_svr_model.sav'
    joblib.dump(svr_model, model_name)
    y_pred, y_test = svr.make_prediction_svr(svr_model, test_set, scaler)
    return y_pred, y_test

def lstm_for_combining(ratio, resid_set, n_lags, hidden_nodes, epochs):
    """Building and making prediction with lstm model."""
    resid_set = pd.Series(resid_set)
    resid_set = resid_set.values
    resid_set = resid_set.reshape(-1, 1)
    resid_set, scaler = tsfut.scale_time_series(resid_set, ratio=ratio)
    train_set, test_set = tsfut.create_dataset_ann(resid_set, lags=n_lags, \
                                                   ratio=ratio)
    print("Running lstm model for combining...")
    lstm_model = lstm.build_lstm_model(train_set, n_lags, \
                                      hidden_nodes=hidden_nodes, epochs=epochs)
    model_name='models/nlags_%s_hidden_nodes_%s_epochs_%s_' % (n_lags, hidden_nodes,\
                                                     epochs) + 'cb_lstm_model.h5'
    lstm_model.save(model_name)
    y_pred, y_test = lstm.make_prediction_lstm(lstm_model, test_set, scaler)
    return y_pred, y_test

def combine_linear_nonlinear_model(combining_type):
    """Combine arima and ffnn model."""
    if os.path.exists('results/cb_ratio_0.800000_arima_result.txt') == False or \
           os.stat('results/cb_ratio_0.800000_arima_result.txt').st_size == 0:
        print("Train and predict arima model, please!")
    else:
        result = tsfut.read_string_from_file('results/cb_ratio_0.800000_arima_result.txt')
        y_pred = list()
        y_test = list()
        resid_set = list()
        for r in result:
            values = r.split(",")
            y_pred.append(float(values[0]))
            y_test.append(float(values[1]))
            resid_set.append(float(values[2]))
        if combining_type == 'arima_ffnn':
            ratio = 0.5
            n_lags = 2
            hidden_nodes = 4
            epochs = 20
            size = int(len(y_pred)*ratio)
            y_pred_ffnn,y_test_ffnn=ffnn_for_combining(ratio,resid_set,\
                                                    n_lags,hidden_nodes,epochs)
            y_pred_arima = y_pred[size:]
            y_pred_arima = np.asarray(y_pred_arima)
            y_test = y_test[size:]
            y_pred_cb = y_pred_arima + y_pred_ffnn
            rmse = tsfut.root_mean_squared_error(y_pred_cb, y_test)
            mape = tsfut.mean_absolute_percentage_error(y_pred_cb, y_test)
            print('CB_ARIMA_FFNN - RMSE: %f, MAPE: %f' % (rmse, mape) + '%')
        elif combining_type == 'arima_cnn':
            ratio = 0.5
            n_lags = 3
            hidden_nodes = 8
            epochs = 20
            size = int(len(y_pred)*ratio)
            y_pred_cnn,y_test_cnn=cnn_for_combining(ratio,resid_set,\
                                                    n_lags,hidden_nodes,epochs)
            y_pred_arima = y_pred[size:]
            y_pred_arima = np.asarray(y_pred_arima)
            y_test = y_test[size:]
            y_pred_cb = y_pred_arima + y_pred_cnn
            rmse = tsfut.root_mean_squared_error(y_pred_cb, y_test)
            mape = tsfut.mean_absolute_percentage_error(y_pred_cb, y_test)
            print('CB_ARIMA_CNN - RMSE: %f, MAPE: %f' % (rmse, mape) + '%')
        elif combining_type == 'arima_lstm':
            ratio = 0.5
            n_lags = 1
            hidden_nodes = 6
            epochs = 100
            size = int(len(y_pred)*ratio)
            y_pred_lstm,y_test_lstm=lstm_for_combining(ratio,resid_set,\
                                                    n_lags,hidden_nodes,epochs)
            y_pred_arima = y_pred[size:]
            y_pred_arima = np.asarray(y_pred_arima)
            y_test = y_test[size:]
            y_pred_cb = y_pred_arima + y_pred_lstm
            rmse = tsfut.root_mean_squared_error(y_pred_cb, y_test)
            mape = tsfut.mean_absolute_percentage_error(y_pred_cb, y_test)
            print('CB_ARIMA_LSTM - RMSE: %f, MAPE: %f' % (rmse, mape) + '%')
        elif combining_type == 'arima_svr':
            ratio = 0.5
            n_lags = 1
            size = int(len(y_pred)*ratio)
            y_pred_svr,y_test_svr=svr_for_combining(ratio,resid_set,n_lags)
            y_pred_arima = y_pred[size:]
            y_pred_arima = np.asarray(y_pred_arima)
            y_test = y_test[size:]
            y_pred_cb = y_pred_arima + y_pred_svr
            rmse = tsfut.root_mean_squared_error(y_pred_cb, y_test)
            mape = tsfut.mean_absolute_percentage_error(y_pred_cb, y_test)
            print('CB_ARIMA_SVR - RMSE: %f, MAPE: %f' % (rmse, mape) + '%')
        else:
            print("Choose combining type, please!")