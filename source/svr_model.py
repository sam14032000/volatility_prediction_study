# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 13:22:59 2020

@author: saksh
"""
import numpy as np
np.random.seed(1337)
import tensorflow as tf

import pandas as pd
from matplotlib.pyplot import *

from ml_pap import *

matplotlib.pyplot.style.use('classic')

"""
All data imported and scaled according to s&p500
"""
sp500 = pd.read_csv('^GSPC.csv', header = 0, index_col = 'Date')
sp500.index = pd.to_datetime(sp500.index, format = '%d-%m-%y')
sp500 = sp500[1:]

nifty = pd.read_csv('^NSEI.csv', header = 0, index_col = 'Date')
nifty.index = pd.to_datetime(nifty.index, format = '%d-%m-%y')
nifty = nifty.reindex(index = sp500.index, method = 'bfill')
nifty.fillna(method = 'bfill', inplace=True)

sing_sti = pd.read_csv('^sti_d.csv', header = 0, index_col = 'Date')
sing_sti.index = pd.to_datetime(sing_sti.index, format = '%Y-%m-%d')
sing_sti = sing_sti.reindex(index = sp500.index, method = 'bfill')
sing_sti.fillna(method = 'bfill', inplace=True)

uk_100 = pd.read_csv('^ukx_d.csv', header = 0, index_col = 'Date')
uk_100.index = pd.to_datetime(uk_100.index, format = '%Y-%m-%d')
uk_100 = uk_100.reindex(index = sp500.index, method = 'bfill')
uk_100.fillna(method = 'bfill', inplace=True)

hangseng = pd.read_csv('^hsi_d.csv', header = 0, index_col = 'Date')
hangseng.index = pd.to_datetime(hangseng.index, format = '%Y-%m-%d')
hangseng = hangseng.reindex(index = sp500.index, method = 'bfill')
hangseng.fillna(method = 'bfill', inplace=True)

nikkei = pd.read_csv('^nkx_d.csv', header = 0, index_col = 'Date')
nikkei.index = pd.to_datetime(nikkei.index, format = '%Y-%m-%d')
nikkei = nikkei.reindex(index = sp500.index, method = 'bfill')
nikkei.fillna(method = 'bfill', inplace=True)

shanghai_comp = pd.read_csv('^shc_d.csv', header = 0, index_col = 'Date')
shanghai_comp.index = pd.to_datetime(shanghai_comp.index, format = '%Y-%m-%d')
shanghai_comp = shanghai_comp.reindex(index = sp500.index, method = 'bfill')
shanghai_comp.fillna(method = 'bfill', inplace=True)

df = pd.DataFrame(index = sp500.index)
df['nifty'] = nifty['Close']
df['sing_sti'] = sing_sti['Close']
df['hangseng'] = hangseng['Close']
df['nikkei'] = nikkei['Close']
df['shanghai_comp'] = shanghai_comp['Close']
df['sp500'] = sp500['Close']
df['uk_100'] = uk_100['Close']

data_cache = df.copy() 
data_cache.dropna(inplace = True)

"""
Run this block to include VAR residuals in input data
"""
for label in data_cache.columns[1:]:
    resids = var_resids('nifty', label, data_cache = data_cache)
    data_cache[str.format("nifty_var_%s"%(label))] = resids 
data_cache.dropna(inplace = True)

"""
Set model targets
"""
data_cache = data_cache[-3090:-90]
data_cache['nifty_volatility'] = np.log(data_cache['nifty']/data_cache['nifty'].shift(1))**2
data_cache.dropna(inplace = True)
data_cache['targets'] = data_cache['nifty_volatility'].shift(-1)
data_cache.dropna(inplace = True)

"""
Split datasets
Use "dense" for SVR predictions
Returns sclaer used for scaling the output variables
"""
X_train, X_test, y_train, y_test, output_scaler = make_datasets(data_cache, model_name = 'dense')

"""
Run model
Inverse transform test targets and predictions
"""
result = svr_model(X_train, y_train, {'C' : [1,10]})
y_pred = result.predict(X_test)
y_test = y_test.reshape(len(y_test), 1)
y_pred = y_pred.reshape(len(y_pred), 1)
y_pred = output_scaler.inverse_transform(y_pred)
y_test = output_scaler.inverse_transform(y_test)

"""
RMSE of inverse transformed variables
"""
m = tf.metrics.RootMeanSquaredError()
m.update_state(y_test, np.abs(y_pred))
transformed = m.result().numpy()
print("RMSE of transformed variables: %d", transformed)

df_plot = make_save_plot(data_cache.index, y_test, y_pred)