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
from .common_tools import  timing,datestr2int
    


@timing
def get_daily_from_pandas(ts_code, ts_market, start_datestr, end_datestr):
    
    market_map = {"sh":"SS", "sz":"SZ"}
    selected_columns = ['date','open', 'high', 'close', 'low', 'volume']
    
    start_year, start_month, start_day = datestr2int(start_datestr)
    end_year, end_month, end_day = datestr2int(end_datestr)
    
    
    query_code_name  = ".".join([ts_code, market_map[ts_market]])
    df = web.DataReader(query_code_name, "yahoo", datetime.datetime(start_year, start_month, start_day ), 
                               datetime.datetime(end_year, end_month, end_day ))
    
    df = df.reset_index()
    
    df = df.rename(columns={"Date":"date","Open":"open", "Close":"close","High":"high","Low":"low", "Volume":"volume"})
    df = df[selected_columns]
    df.volume = df.volume.astype("int")
    return df


@timing
def get_daily_from_tushare_old(ts_code,ts_market,start_datestr, end_datestr):
    selected_columns = ['date','open', 'close', 'high', 'low', 'volume']
    df = ts.get_hist_data(ts_code, start = start_datestr, end = end_datestr, retry_count=3, pause = 1e-3, ktype = "D")
    df = df.reset_index()
    
    df = df[selected_columns]
    df = df.sort_values(by="date", axis=0, ascending=True)
    df.reset_index(inplace=True, drop= True)
    df.date = pd.to_datetime(df.date)
    df.volume = df.volume * 100
    df.volume = df.volume.astype("int")
    
    return df


@timing
def get_daily_from_baostock(ts_code,ts_market,start_datestr, end_datestr):
    lg = bs.login()

    print('login respond error_code:'+lg.error_code)
    print('login respond  error_msg:'+lg.error_msg)
    
    fields = "date,open,close,high,low,volume"
    
    query_code_name  = ".".join([ts_market, ts_code])
    
    # adjustflag：复权类型，默认不复权：3；1：后复权；2：前复权。已支持分钟线、日线、周线、月线前后复权。
    # frequency：数据类型，默认为d，日k线；d=日k线、w=周、m=月、5=5分钟、15=15分钟、30=30分钟、60=60分钟k线数据，不区分大小写
    
    df_query = bs.query_history_k_data(query_code_name, fields, start_date = start_datestr,end_date = end_datestr, frequency= "d")
    
    data_list = []
    while((df_query.error_code =="0") & df_query.next()):
        data_list.append(df_query.get_row_data())
    
    df = pd.DataFrame(data_list, columns = df_query.fields)
    df.date = pd.to_datetime(df.date)
    df.open = df.open.astype("float64")
    df.close = df.close.astype("float64")
    df.high = df.high.astype("float64")
    df.low = df.low.astype("float64")
    df.volume = df.volume.astype("int")
    bs.logout()
    
    return df
    

def main():
    start_datestr, end_datestr = "2020-01-01", "2021-01-01"
    stock_code = "600000"
    stock_market = "sh"  # "sh,sz"

    result1 = get_daily_from_pandas(stock_code,stock_market,start_datestr, end_datestr)

    result2 = get_daily_from_tushare_old(stock_code,stock_market,start_datestr, end_datestr)

    result3 = get_daily_from_baostock(stock_code,stock_market,start_datestr, end_datestr)
    return 0

# %%

if __name__ == '__main__':
    main()



#%%




    