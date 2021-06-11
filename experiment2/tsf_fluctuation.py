# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 06:46:25 2019

@author: ACER
"""
import tsf_utils as tsfut
import lstm
import ffnn
import cnn
import svr
import math
import numpy as np

def compute_error_fluctuation_interval(pred_values, obs_values, fluct_values):
    """Compute error for each fluctuation interval."""
    interval_list=tsfut.define_class_interval(pred_values, \
                obs_values, fluct_values, n_class=12, class_interval=2)
    error_list = list()
    for l in range(len(interval_list)):
        rmse = 0
        mape = 0
        if(len(interval_list[l][0])>0):
            rmse = tsfut.root_mean_squared_error(interval_list[l][0], interval_list[l][1])
            mape = tsfut.mean_absolute_percentage_error(interval_list[l][0], interval_list[l][1])
        print('interval %d: RMSE: %f, MAPE: %f' % (l, rmse, mape), '%')
        error_list.append([rmse, mape])
    return error_list

def fluctuation_arima_model(ratio=0.9):
    """Load and computer error with arima model base on fluctuation.
       ratio: rate of train and test set
    """
    close_btc = tsfut.load_time_series_data()
    train_set, validation_set, test_set, validation_fluct, test_fluct = \
        tsfut.split_time_series_dataset_with_fluctuation(close_btc, ratio=ratio)
    arima_path = 'results/cb_ratio_0.800000_arima_result.txt'
    arima_result = tsfut.read_string_from_file(arima_path)
    arima_pred = list()
    arima_obs = list()
    for r in arima_result:
        values = r.split(",")
        arima_pred.append(float(values[0]))
        arima_obs.append(float(values[1]))
    validation_size = len(validation_fluct)
    validation_pred_values = arima_pred[:validation_size]
    validation_obs_values = arima_obs[:validation_size]
    test_pred_values = arima_pred[validation_size:]
    test_obs_values = arima_obs[validation_size:]
    validation_error_list = compute_error_fluctuation_interval(\
            validation_pred_values, validation_obs_values, validation_fluct)
    return test_pred_values

def fluctuation_ffnn_model(ratio=0.9):
    """Load and compute error with ffnn model base on fluctuation.
       ratio: rate of train and test set
    """
    close_btc = tsfut.load_time_series_data()
    train_set, validation_set, test_set, validation_fluct, test_fluct = \
        tsfut.split_time_series_dataset_with_fluctuation(close_btc,ratio=ratio)
    n_lags = 6
    model_path='models/cb_models/ratio_0.800000_nlags_6_hidden_nodes_12_epochs_100_ffnn_model.h5'
    ffnn_pred,ffnn_obs=ffnn.trained_ffnn_model(ratio=0.8,n_lags=n_lags,model_path=model_path)
    validation_size = len(validation_fluct)
    validation_pred_values = ffnn_pred[:validation_size]
    validation_obs_values = ffnn_obs[:validation_size]
    test_pred_values = ffnn_pred[validation_size:]
    test_obs_values = ffnn_obs[validation_size:]
    validation_error_list = compute_error_fluctuation_interval(\
               validation_pred_values, validation_obs_values, validation_fluct)
    return test_pred_values

def fluctuation_cnn_model(ratio=0.9):
    """Load and compute error with cnn model base on fluctuation.
       ratio: rate of train and test set
    """
    close_btc = tsfut.load_time_series_data()
    train_set, validation_set, test_set, validation_fluct, test_fluct = \
        tsfut.split_time_series_dataset_with_fluctuation(close_btc,ratio=ratio)
    n_lags = 7
    model_path = 'models/cb_models/ratio_0.800000_nlags_7_hidden_nodes_7_epochs_200_cnn_model.h5'
    cnn_pred, cnn_obs = cnn.trained_cnn_model(ratio=0.8,n_lags=n_lags,model_path=model_path)
    validation_size = len(validation_fluct)
    validation_pred_values = cnn_pred[:validation_size]
    validation_obs_values = cnn_obs[:validation_size]
    test_pred_values = cnn_pred[validation_size:]
    test_obs_values = cnn_obs[validation_size:]
    validation_error_list = compute_error_fluctuation_interval(\
               validation_pred_values, validation_obs_values, validation_fluct)
    return test_pred_values

def fluctuation_lstm_model(ratio=0.9):
    """Load and compute error with cnn model base on fluctuation.
       ratio: rate of train and test set
    """
    close_btc = tsfut.load_time_series_data()
    train_set, validation_set, test_set, validation_fluct, test_fluct = \
        tsfut.split_time_series_dataset_with_fluctuation(close_btc,ratio=ratio)
    n_lags = 1
    model_path='models/cb_models/ratio_0.900000_nlags_1_hidden_nodes_100_epochs_10_lstm_model.h5'
    lstm_pred, lstm_obs = lstm.trained_lstm_model(ratio=0.8,n_lags=n_lags,model_path=model_path)
    validation_size = len(validation_fluct)
    validation_pred_values = lstm_pred[:validation_size]
    validation_obs_values = lstm_obs[:validation_size]
    test_pred_values = lstm_pred[validation_size:]
    test_obs_values = lstm_obs[validation_size:]
    validation_error_list = compute_error_fluctuation_interval(\
               validation_pred_values, validation_obs_values, validation_fluct)
    return test_pred_values

def fluctuation_svr_model(ratio=0.9):
    """Load and compute error with cnn model base on fluctuation.
       ratio: rate of train and test set
    """
    close_btc = tsfut.load_time_series_data()
    train_set, validation_set, test_set, validation_fluct, test_fluct = \
        tsfut.split_time_series_dataset_with_fluctuation(close_btc,ratio=ratio)
    n_lags = 6
    model_path='models/cb_models/ratio_0.800000_nlags_6_kernel_rbf_svr_model.sav'
    svr_pred, svr_obs = svr.trained_svr_model(ratio=0.8,n_lags=n_lags,model_path=model_path)
    validation_size = len(validation_fluct)
    validation_pred_values = svr_pred[:validation_size]
    validation_obs_values = svr_obs[:validation_size]
    test_pred_values = svr_pred[validation_size:]
    test_obs_values = svr_obs[validation_size:]
    validation_error_list = compute_error_fluctuation_interval(\
               validation_pred_values, validation_obs_values, validation_fluct)
    return test_pred_values

def combine_models_with_fluctuation(ratio=0.9):
    close_btc = tsfut.load_time_series_data()
    train_set, validation_set, test_set, validation_fluct, test_fluct = \
        tsfut.split_time_series_dataset_with_fluctuation(close_btc,ratio=ratio)
    arima_pred = fluctuation_arima_model(ratio=0.8)
    ffnn_pred = fluctuation_ffnn_model(ratio=0.8)
    cnn_pred = fluctuation_cnn_model(ratio=0.8)
    svr_pred = fluctuation_svr_model(ratio=0.8)
    lstm_pred = fluctuation_lstm_model(ratio=0.8)
    fluct_model = ['arima','arima','arima','svr','lstm','cnn','arima',\
                   'cnn','cnn','arima','lstm','arima']
    y_pred = list()
    n_class=12
    class_interval = 2
    for i in range(len(test_fluct)):
        fluct = math.floor(test_fluct[i])
        order_interval = math.floor(fluct/class_interval)
        if order_interval > (n_class-1):
            order_interval = n_class-1
        model = fluct_model[order_interval]
        if model=='arima':
            y_pred.append(arima_pred[i])
        elif model=='ffnn':
            y_pred.append(ffnn_pred[i])
        elif model=='cnn':
            y_pred.append(cnn_pred[i])
        elif model=='svr':
            y_pred.append(svr_pred[i])
        elif model=='lstm':
            y_pred.append(lstm_pred[i])
        else:
            y_pred.append(lstm_pred[i])
    y_pred = np.asarray(y_pred)
    y_test = test_set.values
    rmse = tsfut.root_mean_squared_error(y_pred, y_test)
    mape = tsfut.mean_absolute_percentage_error(y_pred, y_test)
    print('RMSE: %f, MAPE: %f' % (rmse, mape), '%')
    

    
    