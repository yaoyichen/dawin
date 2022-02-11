#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 23:38:03 2022

@author: yao
"""

import pandas as pd
import os
import numpy as np

# shift span of the data


data_dir = "/home/yao/project/dawin/data/baostock_daily/hasvolume_data"
padding_span = 60
shift_span = 21
feature_span = 120
target_span = 60

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