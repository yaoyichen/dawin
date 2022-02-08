#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 22:53:37 2022

@author: yao
"""
#%%
import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import pandas_ta as ta
import os
import numpy as np



data_dir = "../data/baostock_daily"
df = pd.read_csv(os.path.join(data_dir,f"stock_600000.csv"))

def remove_df_zero_volume(df,column_name):
    indexZeros = df[ df[column_name] == 0 ].index
    df.drop(indexZeros , inplace=True)
    df = df.reset_index(drop = True)
    return df


df = remove_df_zero_volume(df,'volume')

#%%
start_line = 100
end_line = 160
predict_length = 60

input_tensor = torch.tensor(np.asarray(df.iloc[start_line:end_line]['close']))
output_tensor = torch.tensor(np.asarray(df.iloc[end_line:end_line + predict_length]['close']))

gain_tensor = output_tensor/input_tensor[-1]

min_value = torch.min(gain_tensor)
max_value = torch.max(gain_tensor)

target_value = 0
if(min_value>0.95 and max_value > 1.30):
    target_value = 1

print(target_value)


#%%

class BaostockDailyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        # get the stock list and total length
        
        pass

    def __len__(self):
        pass
        return len(self.annotations)

    def __getitem__(self, index):
        return
        
        
        # img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        # image = io.imread(img_path)
        # y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        # if self.transform:
        #     image = self.transform(image)

        # return (image, y_label)
    