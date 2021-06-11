# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 15:03:27 2019

@author: ACER
"""

import pandas as pd
import numpy as np
from itertools import product
import math
import tsf_utils as tsfut
import os
from sklearn.metrics import mean_squared_error
import ffnn
import cnn
import lstm
import svr
from sklearn.preprocessing import MinMaxScaler

def load_ffnn_model():
    """Load trained ffnn model"""
    ratio = 0.9
    n_lags_ffnn = 5
    model_path_ffnn = 'models/single_models/ratio_0.900000_nlags_5_hidden_nodes_10_epochs_200_ffnn_model.h5'
    ffnn_pred,ffnn_obs=ffnn.trained_ffnn_model(ratio,n_lags_ffnn,model_path_ffnn)
    for i in range(len(ffnn_pred)):
        error = (abs(ffnn_pred[i] - ffnn_obs[i]) / ffnn_obs[i]) * 100
        print('predicted = %f,   expected = %f,   error = %f' % (ffnn_pred[i], ffnn_obs[i], error), '%')
    rmse = tsfut.root_mean_squared_error(ffnn_pred, ffnn_obs)
    mape = tsfut.mean_absolute_percentage_error(\
                    np.asarray(ffnn_pred),np.asarray(ffnn_obs))
    print('RMSE: %f, MAPE: %f' % (rmse, mape) + '%')
    
def load_cnn_model():
    """Load trained cnn model"""
    ratio = 0.9
    n_lags_cnn = 7
    model_path_cnn = 'models/single_models/ratio_0.900000_nlags_7_hidden_nodes_14_epochs_200_cnn_model.h5'
    cnn_pred,cnn_obs=cnn.trained_cnn_model(ratio,n_lags_cnn,model_path_cnn)
    for i in range(len(cnn_pred)):
        error = (abs(cnn_pred[i] - cnn_obs[i]) / cnn_obs[i]) * 100
        print('predicted = %f,   expected = %f,   error = %f' % (cnn_pred[i], cnn_obs[i], error), '%')
    rmse = tsfut.root_mean_squared_error(cnn_pred, cnn_obs)
    mape = tsfut.mean_absolute_percentage_error(\
                    np.asarray(cnn_pred),np.asarray(cnn_obs))
    print('RMSE: %f, MAPE: %f' % (rmse, mape) + '%')

def load_lstm_model():
    """Load trained lstm model"""
    ratio = 0.9
    n_lags_lstm = 1
    model_path_lstm = 'models/single_models/ratio_0.900000_nlags_1_hidden_nodes_100_epochs_10_lstm_model.h5'
    lstm_pred,lstm_obs=lstm.trained_lstm_model(ratio,n_lags_lstm,model_path_lstm)
    for i in range(len(lstm_pred)):
        error = (abs(lstm_pred[i] - lstm_obs[i]) / lstm_obs[i]) * 100
        print('predicted = %f,   expected = %f,   error = %f' % (lstm_pred[i], lstm_obs[i], error), '%')
    rmse = tsfut.root_mean_squared_error(lstm_pred, lstm_obs)
    mape = tsfut.mean_absolute_percentage_error(\
                    np.asarray(lstm_pred),np.asarray(lstm_obs))
    print('RMSE: %f, MAPE: %f' % (rmse, mape) + '%')

def load_svr_model():
    """Load trained svr model"""
    ratio = 0.9
    n_lags_svr = 4
    model_path_svr = 'models/single_models/ratio_0.900000_nlags_4_kernel_rbf_svr_model.sav'
    svr_pred,svr_obs=svr.trained_svr_model(ratio,n_lags_svr,model_path_svr)
    for i in range(len(svr_pred)):
        error = (abs(svr_pred[i] - svr_obs[i]) / svr_obs[i]) * 100
        print('predicted = %f,   expected = %f,   error = %f' % (svr_pred[i], svr_obs[i], error), '%')
    rmse = tsfut.root_mean_squared_error(svr_pred, svr_obs)
    mape = tsfut.mean_absolute_percentage_error(\
                    np.asarray(svr_pred),np.asarray(svr_obs))
    print('RMSE: %f, MAPE: %f' % (rmse, mape) + '%')

def read_string_from_file(path):
    str_list = []
    with open(path, 'r') as filehandle:  
        for line in filehandle:
            currentPlace = line[:-1]
            str_list.append(currentPlace)
    return str_list

#str_list = print(read_string_from_file('results/fluct_ratio_0.800000_arima_result.txt'))
#print(type(str_list))
##for i in range(len(str_list)):
##    print(str_list[i])

