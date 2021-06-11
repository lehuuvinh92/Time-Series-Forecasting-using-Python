# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 15:47:21 2018

@author: ACER
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math
import datetime as dt
from sklearn.preprocessing import MinMaxScaler

class Config:
    datasetName = "bitcoin"
    ratio = 0.9
    epoch = 200
    learning_rate = 0.05
    dataset_path = "dataset/btc_01102013_08062019.csv"
    save_images_path = "images"

def load_time_series_data():
    """Load and draw chart of dataset."""
    conf = Config()
    data = pd.read_csv(conf.dataset_path)
    date_btc = data['Date']
    date_btc = [date[:10] for date in date_btc]
    date_btc = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in date_btc]
    close_btc = data['Close']
    plt.plot(date_btc, close_btc)
    plt.ylabel('Bitcoin Price in USD')
    plt.show()
    return close_btc

def split_time_series_dataset(time_series, ratio=0.9):
    """Split time series dataset into train_set, test_set.
       time_series: a serie of value
       ratio: rate to split train_set, test_set
    """
    size = int(len(time_series)*ratio)
    train_set = time_series[:size]
    test_set = time_series[size:]
    return train_set, test_set

def split_time_series_dataset_with_fluctuation(time_series, ratio=0.9):
    """Split time series dataset into train_set, validation_set, test_set.
       time_series: a serie of value
       ratio: rate to split train_set, validation_set, test_set
    """
    train_index = int(len(time_series)*(ratio-0.1))
    validation_index = int(len(time_series)*ratio)
    test_index = int(len(time_series)*(ratio+0.1))
    train_set = time_series[:train_index]
    validation_set = time_series[train_index:validation_index]
    test_set = time_series[validation_index:test_index]
    validation_fluct = compute_fluctuation_2_point_before_1_point(\
                                time_series[train_index-2:validation_index])
    test_fluct = compute_fluctuation_2_point_before_1_point(\
                                time_series[validation_index-2:test_index])
    return train_set, validation_set, test_set, validation_fluct, test_fluct

def scale_time_series(time_series, ratio=0.9):
    """Scale time series with MinMaxScaler."""
    size = int(len(time_series)*ratio)
    scaler = MinMaxScaler()
    train_set = time_series[:size]
    test_set = time_series[size:]
    scaled_train_set = scaler.fit_transform(train_set)
    scaled_test_set = scaler.transform(test_set)
    scaled_time_series = np.append(scaled_train_set, scaled_test_set)
    return scaled_time_series, scaler

def create_dataset_ann(time_series, lags=1, ratio=0.9):
    """Create time series dataset for neural network.
       time_series: a serie of value
       lag: array of values as input for neural network
    """
    size = int(len(time_series)*ratio)
    train_set = list()
    test_set = list()
    #Train set
    for x in range(lags, size):
        prev_x = list()
        for i in range(x-lags, x):
            prev_x.append(time_series[i])
        train_set.append([prev_x, time_series[x]])
    #Test set
    for x in range(size, len(time_series)):
        prev_x = list()
        for i in range(x-lags, x):
            prev_x.append(time_series[i])
        test_set.append([prev_x, time_series[x]])
    return train_set, test_set

def prepare_numpy_array_ann(dataset):
    """Prepare input as numpy array for neural network.
       dataset: train_set, test_set for ann
    """
    x = list()
    y = list()
    for el in dataset:
        x.append(el[0])
        y.append(el[1])
    x = np.asarray(x)
    y = np.asarray(y)
    return x,y

def root_mean_squared_error(y_pred, y_true):
    """root-mean-square error (RMSE)."""
    return math.sqrt(mean_squared_error(y_true, y_pred));

def mean_absolute_percentage_error(y_pred, y_true):
    """mean absolute percentage error (MAPE)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def compute_fluctuation_2_point(point1, point2):
    """Compute fluctuation between two points in time series."""
    return (np.abs(point1 - point2)/point1)*100

def compute_fluctuation_2_point_before_1_point(time_series):
    """Compute fluctuation between two points before one point in time series."""
    fluct_series = list()
    time_series = time_series.values
    for i in range(2, len(time_series)):
        point1 = time_series[i-2]
        point2 = time_series[i-1]
        fluct = compute_fluctuation_2_point(point1, point2)
        fluct_series.append(fluct)
    return fluct_series

def define_class_interval(y_pred, y_obs, fluct_list, n_class=10, class_interval=1):
    """Define fluctuation with number of class intervals."""
    interval_list = list()
    for i in range(n_class):
        pred_list = list()
        obs_list = list()
        interval_list.append([pred_list, obs_list])
    for i in range(len(fluct_list)):
        fluct = math.floor(fluct_list[i])
        order_interval = math.floor(fluct/class_interval)
        if order_interval > (n_class-1):
            order_interval = n_class-1
        interval_list[order_interval][0].append(y_pred[i])
        interval_list[order_interval][1].append(y_obs[i])
    return interval_list

def write_string_to_file(path, string):
    """Write to file."""
    with open(path, 'a') as filehandle:  
            filehandle.write('%s\n' % string)
            
def read_string_from_file(path):
    str_list = []
    with open(path, 'r') as filehandle:  
        for line in filehandle:
            currentPlace = line[:-1]
            str_list.append(currentPlace)
    return str_list
    