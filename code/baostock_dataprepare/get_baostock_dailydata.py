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
from tools.staticinfo_tools import get_hs300_list
from tools.common_tools import  timing,check_valid_date


def test1():
    """
    下载沪深300的数据, 全量可能更多吧。
    """
    hs300_stock_list = get_hs300_list()


    date_folder = "../../data/baostock_daily"
    start_datestr = "2015-01-01"
    end_datestr = "2022-05-28"

    print(hs300_stock_list)

    stock_list = ["sz.399006"]

    # stock_list = ['sz.000538', 'sz.000568', 'sz.000596', 'sz.000625', 'sz.000651', 'sz.000661', 'sz.000703', 'sz.000708', 'sz.000725', 'sz.000768', 'sz.000776', 'sz.000783', 'sz.000786', 'sz.000800', 'sz.000858', 'sz.000876', 'sz.000895', 'sz.000938', 'sz.000963', 'sz.000977', 'sz.001979', 'sz.002001', 'sz.002007', 'sz.002008', 'sz.002024', 'sz.002027', 'sz.002032', 'sz.002044', 'sz.002049', 'sz.002050', 'sz.002064', 'sz.002120', 'sz.002129', 'sz.002142', 'sz.002157', 'sz.002179', 'sz.002202', 'sz.002230', 'sz.002236', 'sz.002241', 'sz.002252', 'sz.002271', 'sz.002304', 'sz.002311', 'sz.002352', 'sz.002371', 'sz.002410', 'sz.002414', 'sz.002415', 'sz.002459', 'sz.002460', 'sz.002466', 'sz.002475', 'sz.002493', 'sz.002555', 'sz.002568', 'sz.002594', 'sz.002600', 'sz.002601', 'sz.002602', 'sz.002607', 'sz.002624', 'sz.002709', 'sz.002714', 'sz.002736', 'sz.002791', 'sz.002812', 'sz.002821', 'sz.002841', 'sz.002916', 'sz.002938', 'sz.003816', 'sz.300003', 'sz.300014', 'sz.300015', 'sz.300033', 'sz.300059', 'sz.300122', 'sz.300124', 'sz.300142', 'sz.300144', 'sz.300274', 'sz.300316', 'sz.300347', 'sz.300408', 'sz.300413', 'sz.300433', 'sz.300450', 'sz.300498', 'sz.300529', 'sz.300558', 'sz.300595', 'sz.300601', 'sz.300628', 'sz.300676', 'sz.300677', 'sz.300750', 'sz.300759', 'sz.300760', 'sz.300782', 'sz.300866', 'sz.300888', 'sz.300896', 'sz.300999']

    download_daily_data(stock_list = stock_list,
                        data_folder = date_folder,
                        start_datestr = start_datestr,
                        end_datestr = end_datestr)

    pass 


#%%
def test2():
    """
    测试15分钟类型数据的获取,没有问题的
    """
    start_datestr = "2017-01-01"
    end_datestr = "2022-05-28"
    stock_code = "cyb"
    stock_market = None
    frequency = "d"

    df = get_daily(stock_code,stock_market,
        start_datestr, end_datestr,frequency= frequency, api_platform= "tushare_old")

    date_folder = "../../data/baostock_15"

    df.to_csv(os.path.join(date_folder,f"stock_{stock_code}.csv"),index = False)

    pass


test2()


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
