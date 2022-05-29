#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 23:24:16 2022

@author: yao
"""

import os
import pandas as pd

"""
为了去除掉baostock中, 存在
"""

def remove_df_zero_volume(df, column_name):
    """
    去掉df中列名column_name 为 0 的行
    """
    indexZeros = df[ df[column_name] == 0 ].index
    df.drop(indexZeros , inplace=True)
    df = df.reset_index(drop = True)
    return df




def test():
    data_dir = "/Users/eason/project/dawin/data/baostock_daily/"

    file_list = os.listdir(os.path.join(data_dir))

    for file in file_list[0:3]:
        print(file)
        df = pd.read_csv(os.path.join(data_dir,file))
        df = remove_df_zero_volume(df,"volume")
        df.to_csv(os.path.join(data_dir,"withvolume_data",file))
    

if __name__ == "__main__":
    test()