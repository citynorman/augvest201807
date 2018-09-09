#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 21:04:50 2018

@author: deepmind
"""
import numpy as np
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
pd.core.common.is_list_like = pd.api.types.is_list_like

import pandas_datareader.data as web

import datetime
import seaborn as sns

cfg_dt_begin = datetime.datetime(2002, 1, 1)
cfg_dt_end = datetime.datetime.today()

dfcass = web.DataReader(['FRGSHPUSM649NCIS', 'FRGEXPUSM649NCIS'], 'fred', cfg_dt_begin, cfg_dt_end)

def prepdf(dfg):
    dfg = dfg.reset_index()
    dfg['DATE_PUB'] = dfg['DATE']+pd.DateOffset(days=14)
    dfg['DATE_M'] = dfg['DATE'].dt.month
    dfg['DATE_YQ'] = (dfg['DATE'].dt.year).astype(str)+'Q'+(dfg['DATE'].dt.quarter).astype(str)
    return dfg

df1 = dfcass.pct_change()
df1 = prepdf(df1)

df12 = dfcass.pct_change(12)
df12 = prepdf(df12)

# analyze seasonality
df1.groupby(['DATE_M']).mean()
df12.groupby(['DATE_M']).mean()

sns.boxplot(x='DATE_M',y='FRGSHPUSM649NCIS',data=df1)
sns.boxplot(x='DATE_M',y='FRGSHPUSM649NCIS',data=df12)


df12m = df12.groupby('DATE_YQ').mean()

'''
TODO:
- predict sales growth using df12[['FRGSHPUSM649NCIS', 'FRGEXPUSM649NCIS']]
    - just before announcement df_act['date_announce']
    - predict sales growth as of every df12['DATE_PUB']
- objective: maximize f1 score of rev_isbeat vs rev_isbeat_pred

'''


dfpred = df12m.merge(dfwernyoy, on=['DATE_YQ'])
assert dfpred.groupby('DATE_YQ').size().max()==1

dfpred = dfpred.dropna()
dfpred[['FRGSHPUSM649NCIS', 'FRGEXPUSM649NCIS', 'rev','rev_permile']].corr()

dfpred.plot.scatter('FRGEXPUSM649NCIS','rev')

'''
Q:
    add both as variables? => CV testing
    
'''

import sklearn.linear_model
import sklearn.metrics
from sklearn.model_selection import cross_validate

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


-cross_validate(sklearn.linear_model.LinearRegression(), dfpred[['FRGEXPUSM649NCIS']].values, dfpred['rev'].values, return_train_score=False, scoring=('r2', 'neg_mean_squared_error'), cv=10)['test_neg_mean_squared_error'].mean()
-cross_validate(sklearn.linear_model.LinearRegression(), dfpred[['FRGSHPUSM649NCIS', 'FRGEXPUSM649NCIS']].values, dfpred['rev'].values, return_train_score=False, scoring=('r2', 'neg_mean_squared_error'), cv=10)['test_neg_mean_squared_error'].mean()


