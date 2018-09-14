#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 21:04:50 2018

@author: deepmind
"""
import datetime
import os
import numpy as np
import pandas as pd
pd.set_option('display.expand_frame_repr', False)

import seaborn as sns

import fastparquet

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
# legal disclaimer: illustrative only, not to be used for investments
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1


#****************************************
# get alt data
#****************************************

fnames=['data/financials/cass.pq']
if not all([os.path.exists(f) for f in fnames]):

    pd.core.common.is_list_like = pd.api.types.is_list_like
    import pandas_datareader.data as web
    cfg_dt_begin = datetime.datetime(2002, 1, 1)
    cfg_dt_end = datetime.datetime.today()
    
    dfcass = web.DataReader(['FRGSHPUSM649NCIS', 'FRGEXPUSM649NCIS'], 'fred', cfg_dt_begin, cfg_dt_end)
    fastparquet.write('data/financials/cass.pq',dfcass)

dfcass = fastparquet.ParquetFile('data/financials/cass.pq').to_pandas()

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

#****************************************
# analyze seasonality
#****************************************

df1.groupby(['DATE_M']).mean()
df12.groupby(['DATE_M']).mean()

sns.boxplot(x='DATE_M',y='FRGSHPUSM649NCIS',data=df1)
sns.boxplot(x='DATE_M',y='FRGSHPUSM649NCIS',data=df12)


df12m = df12.groupby('DATE_YQ').mean()

#****************************************
#****************************************
# results before announcement
#****************************************
#****************************************

#****************************************
# preprocessing
#****************************************

# get consensus before announcement
df_est_announce = df_est.groupby('date_fq').tail(1)
df_est_announce = df_est_announce.merge(df_act[['date_fq','date_announce']], on=['date_fq'])
assert (df_est_announce['date_broker_end']<=df_est_announce['date_announce']).all()

# add alt data
df_est_announce['DATE_YQ'] = (df_est_announce['date_fq'].dt.year).astype(str)+'Q'+(df_est_announce['date_fq'].dt.quarter).astype(str)
df_est_announce = df_est_announce.merge(df12m, on=['DATE_YQ'], how='left')
assert df_est_announce.isna().sum().sum()==0

# correlate?
df_est_announce[['rev_yoy','rev_yoy_est','FRGSHPUSM649NCIS', 'FRGEXPUSM649NCIS']].corr()

df_est_announce.plot.scatter('FRGEXPUSM649NCIS','rev_yoy')
df_est_announce.plot.scatter('rev_yoy_est','rev_yoy')

# model
import sklearn.linear_model
import sklearn.metrics
from sklearn.metrics import mean_squared_error, r2_score, f1_score


# simple model
m = sklearn.linear_model.LinearRegression()
r = m.fit(df_est_announce[['FRGEXPUSM649NCIS']], df_est_announce['rev_yoy'])
df_est_announce['rev_yoy_pred_ins'] = m.predict(df_est_announce[['FRGEXPUSM649NCIS']])

df_est_announce[['rev_yoy','rev_yoy_est','rev_yoy_pred_ins', 'FRGEXPUSM649NCIS']].corr()

df_est_announce['is_beat'] = df_est_announce['rev_yoy']>df_est_announce['rev_yoy_est']
df_est_announce['is_beat_pred'] = df_est_announce['rev_yoy_pred_ins']>df_est_announce['rev_yoy_est']
print(f1_score(df_est_announce['is_beat'],df_est_announce['is_beat_pred']))
pd.crosstab(df_est_announce['is_beat'],df_est_announce['is_beat_pred'])

# out-sample
def run_ols(dftrain,dftest):
    if dftrain.shape[0]<4: return np.nan
    m = sklearn.linear_model.LinearRegression()
    r = m.fit(dftrain[['FRGEXPUSM649NCIS']], dftrain['rev_yoy'])
    return m.predict(dftest['FRGEXPUSM649NCIS'])[0]
    
df_pred_ols = []
for iper in range(0,df_est_announce.shape[0]):
    dftrain = df_est_announce.iloc[0:iper-1,:]
    dftest = df_est_announce.iloc[iper,:]
    pred = run_ols(dftrain,dftest)
    df_pred_ols.append(pred)

df_est_announce['rev_yoy_pred_os'] = df_pred_ols

df_est_announce[['rev_yoy','rev_yoy_est','rev_yoy_pred_os','rev_yoy_pred_ins', 'FRGEXPUSM649NCIS']].corr()
df_est_announce['is_beat_pred'] = df_est_announce['rev_yoy_pred_os']>df_est_announce['rev_yoy_est']
print(f1_score(df_est_announce['is_beat'],df_est_announce['is_beat_pred']))
pd.crosstab(df_est_announce['is_beat'],df_est_announce['is_beat_pred'])


# add more variables?
from sklearn.model_selection import cross_validate
# in sample error go down?
-cross_validate(sklearn.linear_model.LinearRegression(), df_est_announce[['FRGEXPUSM649NCIS']].values, df_est_announce['rev_yoy'].values, return_train_score=False, scoring=('r2', 'neg_mean_squared_error'), cv=10)['test_neg_mean_squared_error'].mean()
-cross_validate(sklearn.linear_model.LinearRegression(), df_est_announce[['FRGSHPUSM649NCIS', 'FRGEXPUSM649NCIS']].values, df_est_announce['rev_yoy'].values, return_train_score=False, scoring=('r2', 'neg_mean_squared_error'), cv=10)['test_neg_mean_squared_error'].mean()

# todo: try yipit data
