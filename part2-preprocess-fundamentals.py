#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 11:35:57 2018

@author: deepmind
"""
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
import fastparquet

# => doesn't work
# df_act = fastparquet.ParquetFile('https://s3-us-west-2.amazonaws.com/datasci-finance/data/wern-act-raw.pq').to_pandas()

# => ideal
# d6tpipe._('citynorman-augvest-2018-pt2').files.get({'include':'wern*.pq'}).download('data/financials')

import urllib
urllib.request.urlretrieve('https://s3-us-west-2.amazonaws.com/datasci-finance/data/wern-act-raw.pq', 'data/financials/wern-act-raw.pq')
urllib.request.urlretrieve('https://s3-us-west-2.amazonaws.com/datasci-finance/data/wern-est-raw.pq', 'data/financials/wern-est-raw.pq')

df_act = fastparquet.ParquetFile('data/financials/wern-act-raw.pq').to_pandas()

df_act.columns
df_act['date_fq_yago'] = df_act['date_fq'] - pd.DateOffset(years=1)
df_act = df_act.merge(df_act[['date_fq','rev']], left_on=['date_fq_yago'], right_on=['date_fq'], suffixes=['','_yago'], how='left')
df_act['rev_yoy'] = df_act['rev']/df_act['rev_yago']-1

df_est = fastparquet.ParquetFile('data/financials/wern-est-raw.pq').to_pandas()
df_est = df_est.merge(df_act[['date_fq','rev_yago','rev_yoy']], left_on=['date_fq'], right_on=['date_fq'], suffixes=['','_act'], how='left')
df_est.tail()
df_est['rev_yoy_est'] = df_est['rev_broker']/df_est['rev_yago']-1
df_est['rev_isbeat'] = df_est['rev_yoy']>df_est['rev_yoy_est']

df_est[['rev_yoy_est','rev_yoy']].corr()
(df_est['rev_yoy']-df_est['rev_yoy_est']).abs().median()

df_pred = df_est[[]]