#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 01:09:26 2022

@author: eason
"""

import baostock as bs
import pandas as pd


def get_baostock_stock_base():
    """
    接口文档
    http://baostock.com/baostock/index.php/Python_API%E6%96%87%E6%A1%A3#.E8.AF.81.E5.88.B8.E5.9F.BA.E6.9C.AC.E8.B5.84.E6.96.99.EF.BC.9Aquery_stock_basic.28.29
    type	证券类型，其中1：股票，2：指数，3：其它，4：可转债，5：ETF
    status	上市状态，其中 1：上市，0：退市
    
    """
    # 登陆系统
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:'+lg.error_code)
    print('login respond  error_msg:'+lg.error_msg)
    
    # 获取证券基本资料
    rs = bs.query_stock_basic()
    # rs = bs.query_stock_basic(code_name="浦发银行")  # 支持模糊查询
    # print('query_stock_basic respond error_code:'+rs.error_code)
    # print('query_stock_basic respond  error_msg:'+rs.error_msg)
    
    # 打印结果集
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    # 登出系统
    bs.logout()
    
    return result


#%%
import baostock as bs
import pandas as pd



def get_baostock_industry_info():
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:'+lg.error_code)
    print('login respond  error_msg:'+lg.error_msg)
    
    # 获取行业分类数据
    rs = bs.query_stock_industry()
    # rs = bs.query_stock_basic(code_name="浦发银行")
    # print('query_stock_industry error_code:'+rs.error_code)
    # print('query_stock_industry respond  error_msg:'+rs.error_msg)
    
    # 打印结果集
    industry_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        industry_list.append(rs.get_row_data())
    result = pd.DataFrame(industry_list, columns=rs.fields)
    # 结果集输出到csv文件
    
    # 登出系统
    bs.logout()
    return result


stock_base = get_baostock_stock_base()
stock_base = stock_base[(stock_base["type"] == "1") & (stock_base["status"] == "1")]
stock_industry = get_baostock_industry_info()



stock_base_merge = pd.merge(stock_base[["code","code_name","ipoDate"]],
                            stock_industry[["updateDate","code","industry"]], 
                            left_on = "code", right_on = "code", how = "inner")


#%%
import baostock as bs
import pandas as pd

def download_data(date):
    bs.login()

    # 获取指定日期的指数、股票数据
    stock_rs = bs.query_all_stock(date)
    stock_df = stock_rs.get_data()
    data_df = pd.DataFrame()
    for code in stock_df["code"]:
        print("Downloading :" + code)
        k_rs = bs.query_history_k_data_plus(code, "date,code,open,high,low,close", date, date)
        data_df = data_df.append(k_rs.get_data())
    bs.logout()
    # data_df.to_csv("D:\\demo_assignDayData.csv", encoding="gbk", index=False)
    print(data_df)



# 获取指定日期全部股票的日K线数据
# download_data("2022-06-17")


#%%
# bs.login()
# date = "2022-06-16"
# stock_rs = bs.query_all_stock(date)
# stock_df = stock_rs.get_data()
# data_df = pd.DataFrame()
# for code in stock_df["code"]:
#     print("Downloading :" + code)
#     k_rs = bs.query_history_k_data_plus(code, "date,code,open,high,low,close", date, date)
#     data_df = data_df.append(k_rs.get_data())
# bs.logout()


import akshare as ak

stock_individual_info_em_df = ak.stock_individual_info_em(symbol="000001")
print(stock_individual_info_em_df)