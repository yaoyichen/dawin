#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 21:02:42 2021

@author: yao
"""
from tools.download_data import *
from tools.common_tools import  timing


start_datestr, end_datestr = "2020-01-01", "2021-01-01"
stock_code = "600000"
stock_market = "sh"  # "sh,sz"


result3 = get_daily_from_baostock(stock_code,stock_market,start_datestr, end_datestr)


#%%
import baostock as bs
import pandas as pd

@timing
def get_profit_data_from_baostock(stock_code,stock_market,year, quarter):
    """
    ref: http://baostock.com/baostock/index.php/%E5%AD%A3%E9%A2%91%E7%9B%88%E5%88%A9%E8%83%BD%E5%8A%9B
    
    "netProfit":
    "roeAvg" 净资产收益率(平均)(%)
    """
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:'+lg.error_code)
    print('login respond  error_msg:'+lg.error_msg)
    
    query_code_name  = ".".join([stock_market, stock_code])
    
    # 查询季频估值指标盈利能力
    profit_list = []
    rs_profit = bs.query_profit_data(code= query_code_name, year=2017, quarter=2)
    
    while (rs_profit.error_code == '0') & rs_profit.next():
        profit_list.append(rs_profit.get_row_data())
        
    df = pd.DataFrame(profit_list, columns=rs_profit.fields)
    df["year"] = year
    df["quater"] = quarter
    
    for column_name in ["totalShare","liqaShare"]:
        df[column_name] = df[[column_name]].astype("float").astype("int")

    for column_name in ["netProfit","roeAvg"]:
        df[column_name] = df[[column_name]].astype("float")

    
    # 登出系统
    bs.logout()
    return df



result = get_profit_data_from_baostock(stock_code,stock_market,2020, 4)
print(result)