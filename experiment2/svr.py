# -*- coding: utf-8 -*-
"""
Created on Sun May 12 20:42:45 2019

@author: ACER
"""
import tsf_utils as tsfut
from sklearn.svm import SVR
from sklearn.externals import joblib

def build_svr_model(train_set, kernel):
    """Build a svr model.
       train_set: train set after normalization
    """
    x_train, y_train = tsfut.prepare_numpy_array_ann(train_set)
    svr_model = SVR(kernel=kernel, C=1e3, gamma=0.002)
    model = svr_model.fit(x_train, y_train)
    return model
    
def make_prediction_svr(model, test_set, scaler):
    """Make predictions for test_set."""
    x_test, y_test = tsfut.prepare_numpy_array_ann(test_set)
    y_pred = model.predict(x_test)
    y_pred = y_pred.reshape(-1, 1)
    y_pred = scaler.inverse_transform(y_pred)
    y_pred = y_pred.reshape(y_pred.shape[0])
    y_test = y_test.reshape(-1, 1)
    y_test = scaler.inverse_transform(y_test)
    y_test = y_test.reshape(y_test.shape[0])
    return y_pred, y_test

def svr_model(ratio, n_lags, kernel):
    """Building and making prediction with lstm model."""
    close_btc = tsfut.load_time_series_data()
    close_btc = close_btc.values
    close_btc = close_btc.reshape(-1, 1)
    close_btc, scaler = tsfut.scale_time_series(close_btc, ratio=ratio)
    train_set, test_set = tsfut.create_dataset_ann(close_btc, lags=n_lags, \
                                                   ratio=ratio)
    print("Running svr model...")
    svr_model = build_svr_model(train_set, kernel)
    model_name='models/svr/ratio_%f_nlags_%s_kernel_' % (ratio, n_lags) + kernel + '_svr_model.sav'
    joblib.dump(svr_model, model_name)
    y_pred, y_test = make_prediction_svr(svr_model, test_set, scaler)
    rmse = tsfut.root_mean_squared_error(y_pred, y_test)
    mape = tsfut.mean_absolute_percentage_error(y_pred, y_test)
    return rmse, mape

def trained_svr_model(ratio, n_lags, model_path):
    """Load trained svr model and make prediction."""
    close_btc = tsfut.load_time_series_data()
    close_btc = close_btc.values
    close_btc = close_btc.reshape(-1, 1)
    close_btc, scaler = tsfut.scale_time_series(close_btc, ratio=ratio)
    train_set, test_set = tsfut.create_dataset_ann(close_btc, lags=n_lags, \
                                                   ratio=ratio)
    svr_model = joblib.load(model_path)
    y_pred, y_test = make_prediction_svr(svr_model, test_set, scaler)
    return y_pred, y_test