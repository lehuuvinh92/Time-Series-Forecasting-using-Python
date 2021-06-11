# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 22:39:30 2019

@author: ACER
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import scipy.optimize as optimize
import ffnn
import cnn
import lstm
import svr
from sklearn.svm import SVR
import tsf_utils as tsfut
from keras.models import load_model
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler

def implement_linear_regression_model(x_train, y_train, x_test, y_test, y_scaler):
    """Implement a linear regression model."""
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred = y_scaler.inverse_transform(y_pred)
    y_pred = y_pred.reshape(y_pred.shape[0])
    rmse = tsfut.root_mean_squared_error(y_pred, y_test)
    mape = tsfut.mean_absolute_percentage_error(y_pred, y_test)
    return rmse, mape

def implement_polynomial_regression_model(x_train, y_train, x_test, y_test, y_scaler):
    """Implement a polynomial regression model."""
    transform_x_train = PolynomialFeatures(degree=2, include_bias=False).\
                                                    fit_transform(x_train)
    model = LinearRegression().fit(transform_x_train, y_train)
    transform_x_test = PolynomialFeatures(degree=2, include_bias=False).\
                                                    fit_transform(x_test)
    y_pred = model.predict(transform_x_test)
    y_pred = y_scaler.inverse_transform(y_pred)
    y_pred = y_pred.reshape(y_pred.shape[0])
    rmse = tsfut.root_mean_squared_error(y_pred, y_test)
    mape = tsfut.mean_absolute_percentage_error(y_pred, y_test)
    return rmse, mape

def implement_support_vector_regression_model(x_train, y_train, x_test, y_test, y_scaler):
    """Implement a support vector regression model."""
    svr_model = SVR(kernel='rbf', C=1e3, gamma=0.002)
    model = svr_model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred = y_pred.reshape(-1, 1)
    y_pred = y_scaler.inverse_transform(y_pred)
    y_pred = y_pred.reshape(y_pred.shape[0])
    rmse = tsfut.root_mean_squared_error(y_pred, y_test)
    mape = tsfut.mean_absolute_percentage_error(y_pred, y_test)
    return rmse, mape

def combine_regression(reg_type):
    #arima result
    arima_path = 'results/cb_ratio_0.800000_arima_result.txt'
    arima_result = tsfut.read_string_from_file(arima_path)
    arima_pred = list()
    arima_obs = list()
    for r in arima_result:
        values = r.split(",")
        arima_pred.append(float(values[0]))
        arima_obs.append(float(values[1]))
    ratio = 0.8
    #ffnn result
    n_lags_ffnn = 6
    model_path_ffnn = 'models/cb_models/ratio_0.800000_nlags_6_hidden_nodes_12_epochs_100_ffnn_model.h5'
    ffnn_pred,ffnn_obs=ffnn.trained_ffnn_model(ratio,n_lags_ffnn,model_path_ffnn)
    #cnn result
    n_lags_cnn = 7
    model_path_cnn = 'models/cb_models/ratio_0.800000_nlags_7_hidden_nodes_7_epochs_200_cnn_model.h5'
    cnn_pred, cnn_obs = cnn.trained_cnn_model(ratio,n_lags_cnn,model_path_cnn)
    #svr result
    n_lags_svr = 6
    model_path_svr = 'models/cb_models/ratio_0.800000_nlags_6_kernel_rbf_svr_model.sav'
    svr_pred, svr_obs = svr.trained_svr_model(ratio,n_lags_svr,model_path_svr)
    #lstm result
    n_lags_lstm = 1
    model_path_lstm = 'models/cb_models/ratio_0.900000_nlags_1_hidden_nodes_100_epochs_10_lstm_model.h5'
    lstm_pred, lstm_obs = lstm.trained_lstm_model(ratio,n_lags_lstm,model_path_lstm)
    #split validation and test set for regression
    dataset = list()
    for i in range(len(ffnn_pred)):
        temp = list()
        temp.append(arima_pred[i])
        temp.append(ffnn_pred[i])
        temp.append(cnn_pred[i])
        temp.append(svr_pred[i])
        temp.append(lstm_pred[i])
        dataset.append([temp, ffnn_obs[i]])
    val_ratio = 0.5
    size = int(len(ffnn_pred)*val_ratio)
    train_set = dataset[:size]
    test_set = dataset[size:]
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    x_train, y_train = tsfut.prepare_numpy_array_ann(train_set)
    x_train = x_scaler.fit_transform(x_train)
    y_train = y_train.reshape(-1, 1)
    y_train = y_scaler.fit_transform(y_train)
    x_test, y_test = tsfut.prepare_numpy_array_ann(test_set)
    x_test = x_scaler.transform(x_test)
    if reg_type == 'linear':
        rmse, mape=implement_linear_regression_model(x_train, y_train, x_test,\
                                                     y_test, y_scaler)
        print('Linear Regression - RMSE: %f, MAPE: %f' % (rmse, mape) + '%')
    elif reg_type == 'poly':
        rmse,mape=implement_polynomial_regression_model(x_train, y_train, \
                                                    x_test, y_test, y_scaler)
        print('Poly Regression - RMSE: %f, MAPE: %f' % (rmse, mape) + '%')
    elif reg_type == 'svr':
        rmse,mape=implement_support_vector_regression_model(x_train, y_train, \
                                                    x_test, y_test, y_scaler)
        print('SVRegression - RMSE: %f, MAPE: %f' % (rmse, mape) + '%')
    else:
        print("Choose regression type, please!")
    