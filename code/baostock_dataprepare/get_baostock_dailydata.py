#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 22:25:44 2022

@author: yao
"""
#%%
#用于下载 hs300数据
import os
os.sys.path.append("..")
from tools.download_tools import *
from tools.staticinfo_tools import get_stocklist
from tools.common_tools import  timing,check_valid_date


def test1():
    """
    下载沪深300的数据, 全量可能更多吧。
    """
    stock_list = get_stocklist(stock_type= "a")


    date_folder = "../../data/baostock_daily/20220620_download"
    start_datestr = "2019-01-01"
    end_datestr = "2022-06-20"

    print(stock_list)
    # for  index, stock_name  in enumerate(stock_list):
    #     if(stock_name == "sz.003025" ):
    #         print(index, stock_name)
        


    # exit()

    # stock_list = ["sh.688150"]

    # stock_list = ['sz.000538', 'sz.000568', 'sz.000596', 'sz.000625', 'sz.000651', 'sz.000661', 'sz.000703', 'sz.000708', 'sz.000725', 'sz.000768', 'sz.000776', 'sz.000783', 'sz.000786', 'sz.000800', 'sz.000858', 'sz.000876', 'sz.000895', 'sz.000938', 'sz.000963', 'sz.000977', 'sz.001979', 'sz.002001', 'sz.002007', 'sz.002008', 'sz.002024', 'sz.002027', 'sz.002032', 'sz.002044', 'sz.002049', 'sz.002050', 'sz.002064', 'sz.002120', 'sz.002129', 'sz.002142', 'sz.002157', 'sz.002179', 'sz.002202', 'sz.002230', 'sz.002236', 'sz.002241', 'sz.002252', 'sz.002271', 'sz.002304', 'sz.002311', 'sz.002352', 'sz.002371', 'sz.002410', 'sz.002414', 'sz.002415', 'sz.002459', 'sz.002460', 'sz.002466', 'sz.002475', 'sz.002493', 'sz.002555', 'sz.002568', 'sz.002594', 'sz.002600', 'sz.002601', 'sz.002602', 'sz.002607', 'sz.002624', 'sz.002709', 'sz.002714', 'sz.002736', 'sz.002791', 'sz.002812', 'sz.002821', 'sz.002841', 'sz.002916', 'sz.002938', 'sz.003816', 'sz.300003', 'sz.300014', 'sz.300015', 'sz.300033', 'sz.300059', 'sz.300122', 'sz.300124', 'sz.300142', 'sz.300144', 'sz.300274', 'sz.300316', 'sz.300347', 'sz.300408', 'sz.300413', 'sz.300433', 'sz.300450', 'sz.300498', 'sz.300529', 'sz.300558', 'sz.300595', 'sz.300601', 'sz.300628', 'sz.300676', 'sz.300677', 'sz.300750', 'sz.300759', 'sz.300760', 'sz.300782', 'sz.300866', 'sz.300888', 'sz.300896', 'sz.300999']

    # download_daily_data(stock_list = stock_list[4586::],
    #                     data_folder = date_folder,
    #                     start_datestr = start_datestr,
    #                     end_datestr = end_datestr)
    download_daily_data(stock_list = stock_list[:],
                        data_folder = date_folder,
                        start_datestr = start_datestr,
                        end_datestr = end_datestr,
                        fields = "date,open,close,high,low,volume,turn,tradestatus,pctChg,isST")
    pass 


#%%
def test2():
    """
    测试15分钟类型数据的获取,没有问题的
    """
    start_datestr = "2019-01-01"
    end_datestr = "2022-05-28"
    stock_code = "cyb"
    stock_market = None
    frequency = "d"

    df = get_daily(stock_code,stock_market,
        start_datestr, end_datestr,frequency= frequency, api_platform= "tushare_old")

    date_folder = "../../data/baostock_15"

    df.to_csv(os.path.join(date_folder,f"stock_{stock_code}.csv"),index = False)

    pass


test1()


# def download_daily_data(stock_list, data_folder, start_datestr,end_datestr):
#     """
#     根据 stock_list, 通过baostock接口下载数据, 并存入 date_folder 中
#     """
#     for item in stock_list:
#         stock_market,stock_code = item.split(".")
#         result = get_daily_from_baostock(stock_code,stock_market,start_datestr, end_datestr)
#         data_dir = data_folder
#         result.to_csv(os.path.join(data_dir,f"stock_{stock_code}.csv"),index = False)




# %%

import baostock as bs
import pandas as pd

# 登陆系统
lg = bs.login()
# 显示登陆返回信息
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)

# 查询2015至2017年复权因子
rs_list = []
rs_factor = bs.query_adjust_factor(code="sh.603035", start_date="2015-01-01", end_date="2017-12-31")
while (rs_factor.error_code == '0') & rs_factor.next():
    rs_list.append(rs_factor.get_row_data())
result_factor = pd.DataFrame(rs_list, columns=rs_factor.fields)
# 打印输出
print(result_factor)

# 结果集输出到csv文件
result_factor.to_csv("D:\\adjust_factor_data.csv", encoding="gbk", index=False)

# 登出系统
bs.logout()



#%%
import baostock as bs
import pandas as pd

#### 登陆系统 ####
lg = bs.login()
# 显示登陆返回信息
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)

#### 查询除权除息信息####


# 查询2017年除权除息信息
rs_list = []
rs_dividend_2017 = bs.query_dividend_data(code="sh.603035", year="2022", yearType="report")
while (rs_dividend_2017.error_code == '0') & rs_dividend_2017.next():
    rs_list.append(rs_dividend_2017.get_row_data())
    
    
    
result_dividend = pd.DataFrame(rs_list, columns=rs_dividend_2017.fields)
# 打印输出
print(result_dividend)