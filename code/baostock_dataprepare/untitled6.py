#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 23:24:18 2022

@author: eason
"""

import os
os.sys.path.append("..")
from tools.download_tools import *
from tools.staticinfo_tools import get_stocklist
from tools.common_tools import  timing,check_valid_date



date_folder = "../../data/baostock_daily/20220619_download"

date_folder_write = "../../data/baostock_daily/20220620_make/"

stock_zh_a_spot_em_df, now, query_elapse_time  = get_current_state()
#%%
remove_list = []
datestr = "2022-06-20"
for index, value in stock_zh_a_spot_em_df.iterrows():
    try:
        df_previous = pd.read_csv( os.path.join(date_folder, "stock_" + value["代码"] + ".csv") )
        df_previous = df_previous[["date","open","close","high","low","volume","turn",\
                                   "tradestatus","pctChg","isST"]]
        
        # 
        # date   open  close   high    low    volume turn,tradestatus,pctChg,isST
        df_today = pd.DataFrame.from_dict(data = {
            "index":[df_previous.index[-1] + 1],
                            "date": [datestr],
                            "open":  [value["今开"]],
                            "close": [value["最新价"]],
                            "high":  [value["最高"]],
                            "low":   [value["最低"]],
                            "volume": [int(value["成交量"]*100)],
                            "turn":[value["换手率"]],
                            "tradestatus": 1,
                            "pctChg": [value["涨跌幅"]],
                            "isST" : df_previous["isST"].values[-1]
                            })

        if( not (value["最新价"] > 0) ): 

            last_close = df_previous.loc[df_previous.index[-1],'close']
            df_today = pd.DataFrame.from_dict(data = {
                "index":[df_previous.index[-1] + 1],
                                "date": [datestr],
                                "open":  [last_close],
                                "close": [last_close],
                                "high":  [last_close],
                                "low":   [last_close],
                                "volume": [0],
                                "turn":0.0,
                                "tradestatus":0,
                                "pctChg":0.0,
                                "isST": df_previous["isST"].values[-1]},) 
            
        
        
        df_today = df_today.set_index("index")
        df_merge = pd.concat([df_previous, df_today])
        
        df_merge.to_csv( os.path.join(date_folder_write, "stock_" + value["代码"] + ".csv"),
                        index = False)
       
        
    except:
        print(value["代码"])
        remove_list.append(value["代码"]) 
        
        
        
        
        
        
    
    
    
    