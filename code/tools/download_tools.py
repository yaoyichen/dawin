#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 21:02:42 2021

@author: yao
"""

#%%
import pandas_datareader.data as pandas_datareader_data
import datetime
import baostock as bs
import pandas as pd
import tushare as ts
import os

from .common_tools import  timing, datestr2int
    


@timing
def get_daily_from_pandas(stock_code, stock_market, start_datestr, end_datestr):
    """
    从yahoo获得历史数据
    当晚17点，已经能够查到数据
    """

    market_map = {"sh":"SS", "sz":"SZ"}
    selected_columns = ['date','open', 'high', 'close', 'low', 'volume']
    
    start_year, start_month, start_day = datestr2int(start_datestr)
    end_year, end_month, end_day = datestr2int(end_datestr)
    
    
    query_code_name  = ".".join([stock_code, market_map[stock_market]])
    df = pandas_datareader_data.DataReader(query_code_name, "yahoo", datetime.datetime(start_year, start_month, start_day ), 
                               datetime.datetime(end_year, end_month, end_day ))
    
    df = df.reset_index()
    
    df = df.rename(columns={"Date":"date","Open":"open", "Close":"close","High":"high","Low":"low", "Volume":"volume"})
    df = df[selected_columns]
    df.volume = df.volume.astype("int")
    return df


@timing
def get_daily_from_tushare_old(stock_code,stock_market,start_datestr, end_datestr, frequency = "d"):
    """
    当晚17点，已经能够查到数据

    code：股票代码，即6位数字代码，或者指数代码（sh=上证指数 sz=深圳成指 hs300=沪深300指数 sz50=上证50 zxb=中小板 cyb=创业板）
    start：开始日期，格式YYYY-MM-DD
    end：结束日期，格式YYYY-MM-DD
    ktype：数据类型，D=日k线 W=周 M=月 5=5分钟 15=15分钟 30=30分钟 60=60分钟，默认为D,即获取某天的股票信息
    retry_count：当网络异常后重试次数，默认为3

    """
    ktype_mapping = {"d":"D","w":"W","m":"M", "5":"5" , "15":"15","60":"60"}


    selected_columns = ['date','open', 'close', 'high', 'low', 'volume']
    df = ts.get_hist_data(stock_code, start = start_datestr, end = end_datestr, retry_count=3, pause = 1e-3,
     ktype = ktype_mapping[frequency])
    df = df.reset_index()
    
    df = df[selected_columns]
    df = df.sort_values(by="date", axis=0, ascending=True)
    df.reset_index(inplace=True, drop= True)
    df.date = pd.to_datetime(df.date)
    df.volume = df.volume * 100
    df.volume = df.volume.astype("int")
    return df





@timing
def get_daily_from_baostock(stock_code,stock_market,start_datestr, end_datestr, frequency = "d"):
    """
    从baostock 获得历史数据
    不太确定最新输入在当天被更新的时间
    """
    lg = bs.login()

    print('login respond error_code:'+lg.error_code)
    print('login respond  error_msg:'+lg.error_msg)
    
    fields = "date,open,close,high,low,volume"
    
    query_code_name  = ".".join([stock_market, stock_code])
    
    # adjustflag：复权类型，默认不复权：3；1：后复权；2：前复权。已支持分钟线、日线、周线、月线前后复权。
    # frequency：数据类型，默认为d，日k线；d=日k线、w=周、m=月、5=5分钟、15=15分钟、30=30分钟、60=60分钟k线数据，不区分大小写
    
    df_query = bs.query_history_k_data(query_code_name, fields, start_date = start_datestr,end_date = end_datestr, 
        frequency= frequency)
    
    data_list = []
    while((df_query.error_code =="0") & df_query.next()):
        data_list.append(df_query.get_row_data())
    
    df = pd.DataFrame(data_list, columns = df_query.fields)

    # 如果取得是日线, 则不转换为date数据
    if(frequency == "d"):
        df.date = pd.to_datetime(df.date)
    df.open = df.open.astype("float64")
    df.close = df.close.astype("float64")
    df.high = df.high.astype("float64")
    df.low = df.low.astype("float64")
    df.volume = df.volume.astype("int")
    bs.logout()
    
    return df


def get_daily_from_wangyi(stock_code,stock_market,start_datestr, end_datestr):
    pass


def get_daily(stock_code,stock_market,start_datestr, end_datestr, frequency,
 api_platform = "baostock"):
    if(api_platform == "baostock"):
        # frequency：数据类型，默认为d，日k线；d=日k线、w=周、m=月、5=5分钟、15=15分钟、30=30分钟、60=60分钟k线数据
        df = get_daily_from_baostock(stock_code,stock_market,start_datestr, end_datestr, frequency)
    
    if(api_platform == "tushare_old"):
        df = get_daily_from_tushare_old(stock_code,stock_market,start_datestr, end_datestr, frequency)
    
    if(api_platform == "pandas"):
        if(frequency != "day"):
            print(f"cannot get {frequency} data from pandas dataloader")
        else:
            df = get_daily_from_pandas(stock_code,stock_market,start_datestr, end_datestr, frequency)
    
    return  df 


#%%

def main():
    start_datestr, end_datestr = "2020-01-01", "2022-05-28"
    stock_code = "002468"
    stock_market = "sz"  # "sh,sz"

    result1 = get_daily_from_pandas(stock_code,stock_market,start_datestr, end_datestr)

    result2 = get_daily_from_tushare_old(stock_code,stock_market,start_datestr, end_datestr)

    result3 = get_daily_from_baostock(stock_code,stock_market,start_datestr, end_datestr)
    
    
    return 0

# %%



def download_daily_data(stock_list, data_folder, start_datestr,end_datestr):
    """
    根据 stock_list, 通过baostock接口下载数据, 并存入 date_folder 中
    """
    for item in stock_list:
        stock_market,stock_code = item.split(".")
        result = get_daily_from_baostock(stock_code,stock_market,start_datestr, end_datestr)
        data_dir = data_folder
        result.to_csv(os.path.join(data_dir,f"stock_{stock_code}.csv"),index = False)



# def download_5minute_data(stock_list, data_folder, start_datestr,end_datestr):
#     """
#     根据 stock_list, 通过baostock接口下载数据, 并存入 date_folder 中
#     """
#     for item in stock_list:
#         stock_market,stock_code = item.split(".")
#         result = get_daily_from_baostock(stock_code,stock_market,start_datestr, end_datestr)
#         data_dir = data_folder
#         result.to_csv(os.path.join(data_dir,f"stock_{stock_code}.csv"),index = False)



if __name__ == '__main__':
    main()


#%%




    