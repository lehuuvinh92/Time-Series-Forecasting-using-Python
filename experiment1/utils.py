# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 15:47:21 2018

@author: ACER
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas import Series
from sklearn.metrics import mean_squared_error
import math

class Configuration:
    datasetName = "bitcoin"
    train_size = 0.8
    test_size = 0.2
    epoch = 200
    learning_rate = 0.05
    dataset_path = "dataset/btc.csv"
    save_images_path = "images/temp"

def load_dataset_arima(path, test_size):
    df = pd.read_csv(path, header=None, index_col=0, squeeze=True)
    df_array = df.values
    train_arima, test_arima = train_test_split(df_array, test_size=test_size, shuffle=False)
    return df, train_arima, test_arima

def load_dataset_ann(path, test_size):
    df = pd.read_csv(path, header=0, parse_dates=[0], index_col=0, squeeze=True)
    df_array = df.values
    df_array = fit_dataset(df_array)
    df_array = np.asarray(df_array)
    df_array = df_array.astype('float32')
    x_dataset, y_dataset = df_array[:, :-1], df_array[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size=test_size, shuffle=False)
    return x_train, y_train, x_test, y_test

def fit_dataset(df_array):
    df_array_ts=[]
    for i in range(len(df_array)-1):
        n_row = []
        n_row.extend(df_array[i][:len(df_array[i])-1])
        n_row.extend(df_array[i+1][len(df_array[i+1])-1:])
        df_array_ts.append(n_row)
    return df_array_ts

def root_mean_squared_error(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred));

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def write_list_to_file(path, alist):
    with open(path, 'w') as filehandle:  
        for listitem in alist:
            filehandle.write('%s\n' % listitem)
        
def read_list_from_file(path):
    alist = []
    with open(path, 'r') as filehandle:  
        for line in filehandle:
            currentPlace = line[:-1]
            alist.append(currentPlace)
    return alist
        
def plot_dataset(dataset, title):
    plt.figure(figsize=(16,12))
    plt.title(title)
    plt.plot(dataset, label='Observed', color='#006699')
    plt.show()
    
def plot_prediction(dataset, predictions, title):
    plt.figure(figsize=(16,12))
    plt.title(title)
    plt.plot(dataset, label='Observed', color='#006699')
    plt.plot(predictions, label='Predicted', color='#ff0066')
    plt.show()
    
def plot_bitcoin_dataset(configs):
    series = Series.from_csv(configs.dataset_path)
    series.plot()