#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 21:02:42 2021

@author: yao
"""
#%%
import sys 
sys.path.append("..") 
import os
from tools.download_data import *
from tools.common_tools import  timing,check_valid_date


start_datestr, end_datestr = "2015-01-01", "2022-05-28"
stock_code = "002468"
stock_market = "sz"  # "sh,sz"


result3 = get_daily_from_baostock(stock_code,stock_market,start_datestr, end_datestr)

#%%


#%%
import baostock as bs
import pandas as pd


@timing
def get_profit_data_from_baostock(stock_code,stock_market,year, quarter):
    """
    ref: http://baostock.com/baostock/index.php/%E5%AD%A3%E9%A2%91%E7%9B%88%E5%88%A9%E8%83%BD%E5%8A%9B
    
    "netProfit": -> net_profit     cumsum of this year!!!
    "roeAvg" 净资产收益率(平均)(%)   
    epsTTM -> eps  earn per share:   sum of last 4 quarter 
    totalShare -> total_share
    liqaShare -> tradeble_share
    MBRevenue -> main_business_income
    
    check if all the variables are the same
    """
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:'+lg.error_code)
    print('login respond  error_msg:'+lg.error_msg)
    
    query_code_name  = ".".join([stock_market, stock_code])
    
    # 查询季频估值指标盈利能力
    profit_list = []
    rs_profit = bs.query_profit_data(code= query_code_name, year=year, quarter=quarter)
    
    while (rs_profit.error_code == '0') & rs_profit.next():
        profit_list.append(rs_profit.get_row_data())
        
    df = pd.DataFrame(profit_list, columns=rs_profit.fields)
    df["year"] = year
    df["quater"] = quarter
    
    df = df.rename(columns = {"netProfit":"net_profit", 
                              "epsTTM":"eps",
                              "totalShare":"total_share",
                              "liqaShare":"tradeble_share",
                              "MBRevenue":"main_business_income",
                              "pubDate":"pub_date",
                              "statDate":"stat_date"
                              })
    
    for column_name in ["pub_date","stat_date"]:
        df[column_name] = df[[column_name]].astype("string")
        
    # for column_name in ["total_share","tradeble_share"]:
    #     df[column_name] = df[[column_name]].astype("float").astype("int")
        
    # for column_name in ["net_profit","roeAvg","eps","main_business_income"]:
    #     df[column_name] = df[[column_name]].astype("float")

    
    # 登出系统
    bs.logout()
    return df


stock_code = "600000"
stock_market = "sh"  # "sh,sz"
result = get_profit_data_from_baostock(stock_code,stock_market,2020, 1)
print(result.loc[0])


#%%
import baostock as bs
import pandas as pd

#### 登陆系统 ####
lg = bs.login()
# 显示登陆返回信息
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)

#### 获取公司业绩预告 ####
rs_forecast = bs.query_forecast_report("sh.600000", start_date="2010-01-01", end_date="2017-12-31")
print('query_forecast_reprot respond error_code:'+rs_forecast.error_code)
print('query_forecast_reprot respond  error_msg:'+rs_forecast.error_msg)
rs_forecast_list = []
while (rs_forecast.error_code == '0') & rs_forecast.next():
    # 分页查询，将每页信息合并在一起
    rs_forecast_list.append(rs_forecast.get_row_data())
result_forecast = pd.DataFrame(rs_forecast_list, columns=rs_forecast.fields)
#### 结果集输出到csv文件 ####
result_forecast.to_csv("D:\\forecast_report.csv", encoding="gbk", index=False)
print(result_forecast)

#### 登出系统 ####
bs.logout()



#%%
import baostock as bs
import pandas as pd

# 登陆系统
lg = bs.login()
# 显示登陆返回信息
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)

# 查询杜邦指数
dupont_list = []
rs_dupont = bs.query_dupont_data(code="sh.600000", year=2020, quarter=1)
while (rs_dupont.error_code == '0') & rs_dupont.next():
    dupont_list.append(rs_dupont.get_row_data())
result_profit = pd.DataFrame(dupont_list, columns=rs_dupont.fields)
# 打印输出
print(result_profit)
# 结果集输出到csv文件
result_profit.to_csv("D:\\dupont_data.csv", encoding="gbk", index=False)

# 登出系统
bs.logout()



#%%



    
    
import re,urllib
import xlwt
from bs4 import BeautifulSoup
from time import sleep
import io


def get_profit_from_wangyi(stock_code):
    """
    https://www.jianshu.com/p/99b7f42c61f9
    [https://blog.csdn.net/u014595019/article/details/48445223](https://blog.csdn.net/u014595019/article/details/48445223)
    获得利润表上的信息
    """

    url = 'http://quotes.money.163.com/service/lrb_'+str(stock_code)+'.html'
    while True:
        try:
            content = urllib.request.urlopen(url,timeout=2).read()
            content = content.decode("gbk","ignore")
            
            data = io.StringIO(content)
            df = pd.read_csv(data, sep=",")
            
            df = df.T
            df = df.reset_index()
            
            new_header = df.iloc[0] #grab the first row for the header
            df = df[1:] #take the data less the header row
            df.columns = new_header #set the header row as the df header
            
            
            
            # 1. rename 2. remove unusual datas, 
            df =df.rename(columns= {"报告日期":"stat_date",
                                      "营业总收入(万元)":"total_income",
                                      "利润总额(万元)":"total_profit",
                                      "净利润(万元)":"net_profit"})
            
            selected_fields = ["stat_date","total_income","net_profit"]
            
            df = df[selected_fields]
            
            
            
            df = df[df['stat_date'].apply(check_valid_date) != 0]
            
            
            df.stat_date = pd.to_datetime(df.stat_date)

            for column_name in ["total_income","net_profit"]:
                df[column_name] = df[[column_name]].astype("float64")
            
            
            
            df = df.sort_values(by = ["stat_date"] ,ascending = True)
            df.reset_index(inplace=True, drop= True)
            
            break
        except Exception as e:
            if str(e) =='HTTP Error 404: Not Found':
                break
            else:
                print(e)
                sleep(1)
                continue
    return df





#%%
def get_asset_from_wangyi(stock_code):
    """
    https://www.jianshu.com/p/99b7f42c61f9
    获得资产负债表上的信息
    原始的 固定资产(万元)，负债合计(万元) 
    """

    url = 'http://quotes.money.163.com/service/zcfzb_'+str(stock_code)+'.html'
    while True:
        try:
            content = urllib.request.urlopen(url,timeout=2).read()
            content = content.decode("gbk","ignore")
            
            data = io.StringIO(content)
            df = pd.read_csv(data, sep=",")
            
            
            
            df = df.T
            df = df.reset_index()
            
            new_header = df.iloc[0] #grab the first row for the header
            df = df[1:] #take the data less the header row
            df.columns = new_header #set the header row as the df header
            
            
            
            # 1. rename 2. remove unusual datas, 
            df =df.rename(columns= {"报告日期":"stat_date",
                                      "固定资产(万元)":"fixed_asset",
                                      "负债合计(万元)":"total_debt",
                                      "资产总计(万元)":"total_asset"})
            
            selected_fields = ["stat_date","fixed_asset","total_debt","total_asset"]
            
            df = df[selected_fields]
            
            
            
            df = df[df['stat_date'].apply(check_valid_date) != 0]
            
            
            df.stat_date = pd.to_datetime(df.stat_date)
            
            
            for column_name in ["fixed_asset","total_debt","total_asset"]:
                df[column_name] = df[[column_name]].astype("float64")
            
            df = df.sort_values(by = ["stat_date"] ,ascending = True)
            df.reset_index(inplace=True, drop= True)
            
            df["total_debt"] = df["total_debt"]
            df["total_asset"] = df["total_asset"]
            

            break
        except Exception as e:
            if str(e) =='HTTP Error 404: Not Found':
                break
            else:
                print(e)
                sleep(1)
                continue
                print(e)
                continue
            
    return df

df = get_profit_from_wangyi(600000)
df2 = get_asset_from_wangyi(600000)



#%%
def get_finance_index_from_wangyi(stock_code):
    """
    https://www.jianshu.com/p/99b7f42c61f9

    """
    url = 'http://quotes.money.163.com/service/zycwzb_'+str(stock_code)+'.html?type=report'
    while True:
        try:
            content = urllib.request.urlopen(url,timeout=2).read()
            content = content.decode("gbk","ignore")
            
            data = io.StringIO(content)
            df = pd.read_csv(data, sep=",")
            
            print(df)

            break
        except Exception as e:
            if str(e) =='HTTP Error 404: Not Found':
                break
            else:
                print(e)
                sleep(1)
                continue
    return df

df = get_finance_index_from_wangyi(600000)



#%%
def get_cash_from_wangyi(stock_code):
    """
    https://www.jianshu.com/p/99b7f42c61f9
    获得现金流表上的信息
    """

    url = 'http://quotes.money.163.com/service/xjllb_'+str(stock_code)+'.html'
    while True:
        try:
            content = urllib.request.urlopen(url,timeout=2).read()
            content = content.decode("gbk","ignore")
            
            data = io.StringIO(content)
            df = pd.read_csv(data, sep=",")

            df = df.T
            df = df.reset_index()
            
            new_header = df.iloc[0] #grab the first row for the header
            df = df[1:] #take the data less the header row
            df.columns = new_header #set the header row as the df header
            df.columns = [name.strip() for name in df.columns]
            
            
            # 1. rename 2. remove unusual datas, 
            df =df.rename(columns= {"报告日期":"stat_date",
                                      "现金的期末余额(万元)":"cash_after"})
            
            selected_fields = ["stat_date","cash_after"]
            
            df = df[selected_fields]
            
            
            
            df = df[df['stat_date'].apply(check_valid_date) != 0]
            
            
            df.stat_date = pd.to_datetime(df.stat_date)
            
            
            for column_name in ["cash_after"]:
                df[column_name] = df[[column_name]].astype("float64")
            
            df = df.sort_values(by = ["stat_date"] ,ascending = True)
            df.reset_index(inplace=True, drop= True)
                  
            print(df)

            break
        except Exception as e:
            if str(e) =='HTTP Error 404: Not Found':
                break
            
            else:
                print(e)
                sleep(1)
                continue
    return df

df_asset = get_asset_from_wangyi(600000)
df_cash = get_cash_from_wangyi(600000)
df_profit = get_profit_from_wangyi(600000)


df_finance_1 = pd.merge(df_asset, df_cash, on = ["stat_date"], how = "outer")
df_finance_2 = pd.merge(df_finance_1, df_profit, on = ["stat_date"], how = "outer")

df_finance_2 = df_finance_2.sort_values(by = ["stat_date"] ,ascending = True)
df_finance_2.reset_index(inplace=True, drop= True)