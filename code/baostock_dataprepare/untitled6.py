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



date_folder = "../../data/baostock_daily/20220608"



stock_zh_a_spot_em_df, now, query_elapse_time  = get_current_state()
#%%
remove_list = []
datestr = "2022-06-10"
for index, value in stock_zh_a_spot_em_df.iterrows():
    # print(value)
    # if(value["代码"] == "002468" ):
        # print(
        # value["代码"],
        # value["今开"],
        # value["最新价"],
        # value["最高"],
        # value["最低"],
        # int(value["成交量"]*100),)
        
    
    try:
        df_previous = pd.read_csv( os.path.join(date_folder, "stock_" + value["代码"] + ".csv") )
        
        df_today = pd.DataFrame.from_dict(data = {
            "index":[df_previous.index[-1] + 1],
                            "date": [datestr],
                            "open":  [value["今开"]],
                            "close": [value["最新价"]],
                            "high":  [value["最高"]],
                            "low":   [value["最低"]],
                            "volume": [int(value["成交量"]*100)]},)    
        
        
        df_today = df_today.set_index("index")
        df_merge = pd.concat([df_previous, df_today])
    except:
        print(value["代码"])
        remove_list.append(value["代码"]) 
        
        
        
        
    
    
    
    