# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 13:01:49 2020

@author: saksh
"""
import numpy as np
np.random.seed(1337)
import tensorflow as tf

import pandas as pd
from statsmodels.tsa.api import VAR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from matplotlib.pyplot import *
from datetime import datetime

"""
Calculate VAR residuals. Information criteria for optimal lag = AIC
"""
def var_resids(label1, label2, data_cache):
    model = VAR(data_cache[[label1,label2]])
    model_fit = model.fit(maxlags = 10, ic = 'aic', trend = 'c')
    
    return model_fit.resid[label1]

"""
Data split = 80-10-10
MinMaxScaler applied to both input and output, range = [-1, 1]
LSTM model uses windowing of 3 input steps
"""
def make_datasets(df, target_column = True, train_size = 0.9, model_name = 'dense', input_steps = 3):
    if target_column:
        data = df.iloc[:, :-1]
        data = np.array(data, dtype = np.float32)
        targets = np.array(df.iloc[:,-1], dtype = np.float32)
        X_train, X_test, y_train, y_test = train_test_split(data,targets, train_size = train_size, shuffle = False)
        input_scaler = MinMaxScaler(feature_range = (-1,1))
        input_scaler.fit(X_train)
        X_train = input_scaler.transform(X_train)
        X_test = input_scaler.transform(X_test)
        
        y_train = y_train.reshape(len(y_train), 1)
        y_test = y_test.reshape(len(y_test), 1)
        output_scaler = MinMaxScaler(feature_range = (-1,1))
        output_scaler.fit(y_train)
        y_train = output_scaler.transform(y_train)
        y_test = output_scaler.transform(y_test)
        if model_name == 'dense':
            return X_train, X_test, y_train, y_test, output_scaler
        elif model_name == 'lstm':
            y_train = y_train.reshape(len(y_train), 1)
            input_ds_train = np.hstack((X_train, y_train))
            X_train, y_train = split_sequences(input_ds_train, input_steps)
            
            y_test = y_test.reshape(len(y_test), 1)
            input_ds_test = np.hstack((X_test, y_test))
            X_test, y_test = split_sequences(input_ds_test, input_steps)
            return X_train, X_test, y_train, y_test, output_scaler
    else:
        data = np.array(df, dtype = np.float32)
        X_train, X_test = train_test_split(data, train_size = train_size)
        return X_train, X_test

"""
Early stopping is defined, can be enabled by adding early_stopping to callback
Inputs are batched: batch size = 32
Provides tensorboard accessibility
"""
def nn_model_compile(model, X_train_data, y_train_data, patience = 2, MAX_EPOCHS = 20):
    tf.keras.backend.clear_session()
    model.compile(optimizer = tf.optimizers.SGD(), loss = tf.losses.MeanSquaredError(), metrics = [tf.metrics.RootMeanSquaredError()])
    logdir = "logs\\fit\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min')
    final_res = model.fit(x = X_train_data, y = y_train_data, validation_split = 0.1, epochs = MAX_EPOCHS, batch_size = 32, callbacks=[tensorboard_callback])
    
    return final_res

"""
epsilon = 0.0001
"""
def svr_model(X_train, y_train, param_grid):
    model = GridSearchCV(SVR(epsilon = 0.0001), param_grid, return_train_score=True)
    model.fit(X_train, y_train)
    return model

def split_sequences(input_arr, n_steps):
    X, y = list(), list()
    for i in range(len(input_arr)):
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(input_arr):
            break
        # gather input and output parts of the pattern
        _x, _y = input_arr[i:end_ix, :-1], input_arr[end_ix-1, -1]
        X.append(_x)
        y.append(_y)
    return np.array(X), np.array(y)

def make_save_plot(index, y_test, y_pred, figsize = (6, 6), xlabel = "Date", ylabel = "Market Volatility (Normalized Data)", y_lim = [0.0000, 0.0015], filepath = "default.svg"):
    df_plot = pd.DataFrame(index = index[-len(y_test):])
    df_plot['target_variable'] = y_test
    df_plot['predictions'] = np.abs(y_pred)
    fig, ax = subplots()
    df_plot.plot(figsize=figsize, ax=ax, ylabel = ylabel, xlabel = xlabel)
    ax.legend()
    ax.set_ylim(y_lim)
    savefig(filepath, transparent = True, bbox_inches = 'tight')