# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 04:55:00 2019

@author: ACER
"""
import tsf_utils as tsfut
import arima as ari
import ffnn
import cnn
import lstm
import svr
import tsf_fluctuation as tsfluct
import regression_combining as regcb
import linear_nonlinear_combining as linoncb
import time

def linear_nonlinear_combining_model():
    """Building and making prediction with combining linear and nonlinear model.
       acronym: cblinon
    """
#    combining_type = ['arima_ffnn', 'arima_cnn', 'arima_lstm', 'arima_svr']
    combining_type = ['arima_lstm']
    for cb in combining_type:
        linoncb.combine_linear_nonlinear_model(cb)
    
def fluctuation_combining_model():
    """Building and making prediction with combining models base on fluctuation.
       acronym: cbfluct
    """
    tsfluct.combine_models_with_fluctuation()
    
def regression_combining_model():
    """Building and making prediction with combining models by regression.
       acronym: cbreg
    """
    reg_type = ['linear', 'poly', 'svr']
    for reg in reg_type:
        regcb.combine_regression(reg)
    
def main():
    """Implement models."""
    start = time.time()
    method = 'lstm'
    if method == 'arima':
        ratio = 0.8
        rmse, mape = ari.arima_model(ratio)
        error_str = 'RMSE: %f, MAPE: %f' % (rmse, mape) + '%'
        tsfut.write_string_to_file('results/ratio_%f_arima_result.txt'%(ratio), error_str)
    elif method == 'ffnn':
        ratio = 0.9
        max_lags = 8
        max_hidden_nodes = max_lags * 2
        max_epochs = 11
        param_loops = 1
        for i in range(2, 3):
            for j in range(4, 5):
                for k in range(10, max_epochs, 11):
                    param_str = 'n_lags:%d, hidden_nodes:%d, epochs:%d' % (i, j, k)
                    tsfut.write_string_to_file('results/ratio_%f_ffnn_result.txt'%(ratio), param_str)
                    for l in range(param_loops):
                        n_lags=i
                        hidden_nodes=j
                        epochs=k
                        rmse, mape = ffnn.ffnn_model(ratio, n_lags, hidden_nodes=\
                                                hidden_nodes, epochs=epochs)
                        error_str = 'RMSE: %f, MAPE: %f' % (rmse, mape) + '%'
                        tsfut.write_string_to_file('results/ratio_%f_ffnn_result.txt'%(ratio), error_str)
    elif method == 'cnn':
        ratio = 0.9
        max_lags = 8
        max_hidden_nodes = max_lags * 2
        max_epochs = 11
        param_loops = 1
        for i in range(3, 4):
            for j in range(4, 5):
                for k in range(10, max_epochs, 100):
                    param_str = 'n_lags:%d, hidden_nodes:%d, epochs:%d' % (i, j, k)
                    tsfut.write_string_to_file('results/ratio_%f_cnn_result.txt'%(ratio), param_str)
                    for l in range(param_loops):
                        n_lags=i
                        hidden_nodes=j
                        epochs=k
                        rmse, mape = cnn.cnn_model(ratio, n_lags, hidden_nodes=\
                                                hidden_nodes, epochs=epochs)
                        error_str = 'RMSE: %f, MAPE: %f' % (rmse, mape) + '%'
                        tsfut.write_string_to_file('results/ratio_%f_cnn_result.txt'%(ratio), error_str)
    elif method == 'svr':
        ratio = 0.9
        max_lags = 8
        kernels=['rbf','linear','poly']
        for i in range(1, max_lags):
            for j in kernels:
                param_str = 'n_lags:%d' % i + ', kernel:' + j
                tsfut.write_string_to_file('results/ratio_%f_svr_result.txt'%(ratio), param_str)
                n_lags=i
                kernel=j
                rmse, mape = svr.svr_model(ratio, n_lags, kernel)
                error_str = 'RMSE: %f, MAPE: %f' % (rmse, mape) + '%'
                tsfut.write_string_to_file('results/ratio_%f_svr_result.txt'%(ratio), error_str)
    elif method == 'lstm':
        ratio = 0.9
        max_lags = 6
        max_hidden_nodes = max_lags * 2
        max_epochs = 11
        param_loops = 1
        for i in range(3,4):
            for j in range(100, 101):
                for k in range(100, 201, 100):
                    param_str = 'n_lags:%d, hidden_nodes:%d, epochs:%d' % (i, j, k)
                    tsfut.write_string_to_file('results/ratio_%f_lstm_result.txt'%(ratio), param_str)
                    for l in range(param_loops):
                        n_lags=i
                        hidden_nodes=j
                        epochs=k
                        rmse, mape = lstm.lstm_model(ratio, n_lags, hidden_nodes=\
                                                hidden_nodes, epochs=epochs)
                        error_str = 'RMSE: %f, MAPE: %f' % (rmse, mape) + '%'
                        tsfut.write_string_to_file('results/ratio_%f_lstm_result.txt'%(ratio), error_str)
    elif method == 'cbfluct':
        fluctuation_combining_model()
    elif method == 'cbreg':
        regression_combining_model()
    elif method == 'cblinon':
        linear_nonlinear_combining_model()
    else:
        print("Choose a model, please!")
    end = time.time()
    print(end - start)
if __name__ == "__main__":
	main()