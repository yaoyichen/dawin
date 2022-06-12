#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 01:33:31 2022

@author: eason
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import scipy


data_dir = "../../data/baostock_daily/"


file_list = os.listdir(data_dir)

df_all = pd.DataFrame()

file = "stock_002444.csv"


df = pd.read_csv(os.path.join(data_dir, file))
df = df[df["date"]>= "2017-01-01"]

df["stock_code"] = file.split(".")[0].split("_")[1]
df["increase"] = df["close"]/ df["open"]


result_dict = {}
for index, file2 in enumerate(file_list):
    print(file2)
    if(not file2.endswith(".csv")):
        continue
    df2 = pd.read_csv(os.path.join(data_dir, file2))
    df2 = df2[df2["date"]>= "2017-01-01"]
    
    if(len(df2) <= 20):
        continue
    df2["stock_code"] = file2.split(".")[0].split("_")[1]
    df2["increase"] = df2["close"]/ df2["open"]
    
    tt = pd.merge(df[["date", "stock_code", "increase"]],
                  df2[["date", "stock_code", "increase"]],
                  left_on = ["date"], right_on = ["date"],
                  how = "inner")
                  
    x = np.asarray( tt["increase_x"] - 1)

    y = np.asarray( tt["increase_y"] - 1)
    
    tt = np.dot(x,y) / (np.linalg.norm(x) *np.linalg.norm(y)  )
    
    
    print(file2, tt)
    
    result_dict[file2] = tt
#%%
    
tt = sorted(result_dict.items(),key = lambda x:x[1] ,reverse = True )      