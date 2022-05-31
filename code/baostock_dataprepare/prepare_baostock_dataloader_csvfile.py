#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 23:38:03 2022

@author: yao
"""

import pandas as pd
import os
import numpy as np
import pickle

# shift span of the data


data_dir = "/home/yao/project/dawin/data/baostock_daily/hasvolume_data"

#%%
# 最左边的多少个不取
# padding_span: 左侧多少的数据不取，防止刚上市时候的不稳定
# feature_span: 特征的周期
# target_span: 希望考察的范围

padding_span = 60
shift_span = 10
feature_span = 50
target_span = 5

#时间上前百分之多少的数据是训练集
train_ratio = 0.7

file_list = os.listdir(data_dir)

total_sample_number = 0

code_list = []
feature_start_list = []
target_start_list = []
target_end_list = []


test_code_list = []
test_feature_start_list = []
test_target_start_list = []
test_target_end_list = []


for file in file_list:
    print(file)
    df = pd.read_csv(os.path.join(data_dir, file))
    
    data_len = len(df)
    single_start_list = np.arange(padding_span, data_len - feature_span - target_span,shift_span )
    
    for start_index in single_start_list[0: int(train_ratio*len(single_start_list))]:
        code_list.append(file)
        feature_start_list.append(start_index)
        target_start_list.append(start_index + feature_span)
        target_end_list.append(start_index + feature_span + target_span)
    

    for start_index in single_start_list[int(train_ratio*len(single_start_list)):len(single_start_list)]:
        test_code_list.append(file)
        test_feature_start_list.append(start_index)
        test_target_start_list.append(start_index + feature_span)
        test_target_end_list.append(start_index + feature_span + target_span)
    
print(len(code_list), )

train_df = pd.DataFrame(data={"code":code_list, 
                           "feature_start_index": feature_start_list,
                           "target_start_index":target_start_list,
                           "target_end_index":target_end_list,})

test_df = pd.DataFrame(data={"code":test_code_list, 
                           "feature_start_index": test_feature_start_list,
                           "target_start_index":test_target_start_list,
                           "target_end_index":test_target_end_list,})

train_df.to_csv("baostock_list_train.csv", index = False)
test_df.to_csv("baostock_list_test.csv", index = False)



#%%
start_dict = {}
df_all = pd.DataFrame()
file_list = os.listdir(data_dir)
start_index = 0
for file in file_list:
    print(file)
    df = pd.read_csv(os.path.join(data_dir, file))
    df_all = pd.concat([df_all, df])
    start_dict[file] = [start_index, start_index+len(df)]
    start_index = start_index+ len(df)

dict_file = open("file_index_map.pkl",'wb')
pickle.dump(start_dict, dict_file)    
df_all = df_all.reset_index(drop = True)

df_all.to_csv("baostock_all.csv", index = False)
    
    