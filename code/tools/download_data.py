#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 21:02:42 2021

@author: yao
"""

#%%
import pandas_datareader.data as web
import datetime
import baostock as bs
import pandas as pd
import tushare as ts

def datestr2int(datestr):
    datetime_date = datetime.datetime.strptime(datestr, '%Y-%m-%d')
    return datetime_date.year, datetime_date.month, datetime_date.day


def get_daily_from_pandas(ts_code, ts_market, start_datestr, end_datestr):
    
    market_map = {"sh":"SS", "sz":"SZ"}
    selected_columns = ['date','open', 'high', 'close', 'low', 'volume']
    
    start_year, start_month, start_day = datestr2int(start_datestr)
    end_year, end_month, end_day = datestr2int(end_datestr)
    
    
    query_code_name  = ".".join([ts_code, market_map[ts_market]])
    df_pandas = web.DataReader(query_code_name, "yahoo", datetime.datetime(start_year, start_month, start_day ), 
                               datetime.datetime(end_year, end_month, end_day ))
    
    df_pandas = df_pandas.reset_index()
    
    df_pandas = df_pandas.rename(columns={"Date":"date","Open":"open", "Close":"close","High":"high","Low":"low", "Volume":"volume"})
    df_pandas = df_pandas[selected_columns]
    df_pandas.volume = df_pandas.volume.astype("int")
    return df_pandas



def get_daily_from_tushare_old(ts_code,ts_market,start_datestr, end_datestr):
    selected_columns = ['date','open', 'high', 'close', 'low', 'volume']
    df_tushare_old = ts.get_hist_data(ts_code, start = start_datestr, end = end_datestr, retry_count=3, pause = 1e-3, ktype = "D")
    df_tushare_old = df_tushare_old.reset_index()
    
    df_tushare_old = df_tushare_old[selected_columns]
    df_tushare_old = df_tushare_old.sort_values(by="date", axis=0, ascending=True)
    df_tushare_old.reset_index(inplace=True, drop= True)
    df_tushare_old.date = pd.to_datetime(df_tushare_old.date)
    df_tushare_old.volume = df_tushare_old.volume * 100
    df_tushare_old.volume = df_tushare_old.volume.astype("int")
    
    return df_tushare_old


def get_daily_from_baostock(ts_code,ts_market,start_datestr, end_datestr):
    lg = bs.login()
    
    fields = "date,open,high,close,low,volume"
    
    query_code_name  = ".".join([ts_market, ts_code])
    df_bs_query = bs.query_history_k_data(query_code_name, fields, start_date = start_datestr,end_date = end_datestr, frequency= "d")
    
    data_list = []
    while((df_bs_query.error_code =="0") & df_bs_query.next()):
        data_list.append(df_bs_query.get_row_data())
    
    df_bs = pd.DataFrame(data_list, columns = df_bs_query.fields)
    df_bs.date = pd.to_datetime(df_bs.date)
    df_bs.open = df_bs.open.astype("float64")
    df_bs.close = df_bs.close.astype("float64")
    df_bs.high = df_bs.high.astype("float64")
    df_bs.low = df_bs.low.astype("float64")
    df_bs.volume = df_bs.volume.astype("int")
    
    return df_bs
    
    
start_datestr, end_datestr = "2020-01-01", "2021-01-01"
ts_code = "600000"
ts_market = "sh"  # "sh,sz"

result1 = get_daily_from_pandas(ts_code,ts_market,start_datestr, end_datestr)

result2 = get_daily_from_tushare_old(ts_code,ts_market,start_datestr, end_datestr)

result3 = get_daily_from_baostock(ts_code,ts_market,start_datestr, end_datestr)

# %%




#%%




    