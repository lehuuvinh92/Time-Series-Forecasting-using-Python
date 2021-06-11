# -*- coding: utf-8 -*-
"""
Created on Sun May  6 10:09:25 2018
@author: ACER

"""
import tsf_utils as tsfut
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

def test_stationarity(time_series):
    """Testing the Stationarity with Augmented Dicky Fuller Test."""
    #Determing rolling statistics
    rolmean = time_series.rolling(window=22,center=False).mean()
    rolstd = time_series.rolling(window=12,center=False).std()
    #Plot rolling statistics:
    plt.plot(time_series, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    #Perform Dickey Fuller test    
    result=adfuller(time_series)
    print('ADF Stastistic: %f'%result[0])
    print('p-value: %f'%result[1])
    for key,value in result[4].items():
         if result[0]>value:
            print("The graph is non stationary")
            break
         else:
            print("The graph is stationary")
            break;
    print('Critical values:')
    for key,value in result[4].items():
        print('\t%s: %.3f ' % (key, value))

def identify_parameters_arima(time_series):
    """Identify parametes of ARIMA model."""
    pacf_arr, confint_pacf_arr = pacf(time_series, nlags=32, alpha=.05)
    pacf_arr = pacf_arr[1:]
    confint_pacf_arr = confint_pacf_arr[1:]
    p = list()
    for i in range(0, len(pacf_arr)):
        if pacf_arr[i] < confint_pacf_arr[:, 0][i] - pacf_arr[i] or \
        pacf_arr[i] > confint_pacf_arr[:, 1][i] - pacf_arr[i]:
            p.append(i+1)
    acf_arr, confint_arr  = acf(time_series, nlags=32, alpha=.05)
    acf_arr = acf_arr[1:]
    confint_arr = confint_arr[1:]
    q = list()
    for i in range(0, len(acf_arr)):
        if acf_arr[i] < confint_arr[:, 0][i] - acf_arr[i] or \
        acf_arr[i] > confint_arr[:, 1][i] - acf_arr[i]:
            q.append(i+1)
    return p, q

def select_arima_model(time_series, parameters_list):
    """Select an ARIMA model that is the best compatible with time series."""
    arima_params = []
    warnings.filterwarnings('ignore')
    for param in parameters_list:
        try:
            model = ARIMA(time_series, order=(param[0],param[1],param[2])).fit(disp=-1)
        except:
            print('wrong parameters:', param)
            continue
        arima_params.append([param, model.aic])
    return arima_params

def make_prediction_arima(train_set, test_set, param):
    """Make predictions for test_set.
       train_set: train set after normalization
       test_set: test set after normalization
       param: p, q of the chosen arima model
    """
    history = [x for x in train_set]
    predicted_values = list()
    observed_values = list()
    errored_values = list()
    for t_value in test_set:
        model = ARIMA(history, order=(param[0], param[1], param[2])).fit(disp=-1)
        output = model.forecast()
        pred_value = output[0]
        pred_value = np.exp(pred_value)
        predicted_values.append(float(pred_value))
        obs_value = t_value
        history.append(obs_value)
        obs_value = np.exp(obs_value)
        observed_values.append(float(obs_value))
        # Calculating the error
        error = (abs(pred_value - obs_value) / obs_value) * 100
        errored_values.append(error)
    return predicted_values, observed_values

def arima_model(ratio):
    """Building and making prediction with arima model."""
    close_btc = tsfut.load_time_series_data()
    print(close_btc.describe())
    close_btc.hist()
    train_set, test_set = tsfut.split_time_series_dataset(close_btc,\
                                            ratio=ratio)
    test_stationarity(train_set)
    train_set_log=np.log(train_set)
    test_set_log=np.log(test_set)
    test_stationarity(train_set_log)
    train_set_log_diff = train_set_log - train_set_log.shift()
    train_set_log_diff.dropna(inplace=True)
    test_stationarity(train_set_log_diff)
    model = ARIMA(train_set_log, order=(0,1,5)) 
    results_ARIMA = model.fit(disp=-1)
    print(results_ARIMA.summary())
    p, q = identify_parameters_arima(train_set_log_diff)
    p = p[:3]
    p.insert(0, int(0))
    q = q[:3]
    q.insert(0, int(0))
    d = [1]
    parameters = product(p, d, q)
    parameters_list = list(parameters)
    arima_params = select_arima_model(train_set_log, parameters_list)
    sorted_aic_params = pd.DataFrame(arima_params)
    sorted_aic_params.columns = ['parameters', 'aic']
    sorted_aic_params = sorted_aic_params.sort_values(by = 'aic', ascending=True)
    predicted_values = list()
    observed_values = list()
    print("Running arima model...")
    for index, row in sorted_aic_params.iterrows():
        param = row['parameters']
        try:
            predicted_values, observed_values = make_prediction_arima(\
                                        train_set_log, test_set_log, param)
        except:
            print('Inappropriate model:', param)
            continue
        if predicted_values:
                print("Appropriate parameters:")
                print(param)
                break
    result_path = 'results/ratio_%f_arima_result.txt' % (ratio)
    for i in range(len(predicted_values)):
        result = '%f,%f,%f' % (predicted_values[i], observed_values[i], \
                            float(observed_values[i] - predicted_values[i]))
        tsfut.write_string_to_file(result_path, result)
    rmse = tsfut.root_mean_squared_error(predicted_values, observed_values)
    mape = tsfut.mean_absolute_percentage_error(\
                    np.asarray(predicted_values),np.asarray(observed_values))
    return rmse, mape