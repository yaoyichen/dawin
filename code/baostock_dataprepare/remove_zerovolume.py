#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 23:24:16 2022

@author: yao
"""

import os
import pandas as pd


def remove_df_zero_volume(df,column_name):
    indexZeros = df[ df[column_name] == 0 ].index
    df.drop(indexZeros , inplace=True)
    df = df.reset_index(drop = True)
    return df


data_dir = "/home/yao/project/dawin/data/baostock_daily/"
shift_span = 20

file_list = os.listdir(os.path.join(data_dir,"origin_data"))

for file in file_list:
    df = pd.read_csv(os.path.join(data_dir,"origin_data",file))
    df.to_csv(os.path.join(data_dir,"hasvolume_data",file))
    
    