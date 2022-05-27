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
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle


#%%


def feature_target_1(df, feature_start_index, target_start_index, target_end_index):
    """
    normalize input
    
    """
    input_tensor = torch.tensor(np.asarray(df.iloc[feature_start_index: target_start_index]['close']), dtype = torch.float32)
    volume_tensor = torch.tensor(np.asarray(df.iloc[feature_start_index: target_start_index]['volume']), dtype = torch.float32)
    
    output_tensor = torch.tensor(np.asarray(df.iloc[target_start_index: target_end_index]['close']), dtype = torch.float32)
    
    gain_tensor = output_tensor/input_tensor[-1]
    
    min_value = torch.min(gain_tensor)
    max_value = torch.max(gain_tensor)
    
    target_value = torch.tensor([0.0])
    weight_value = torch.tensor([1.0])
    if(min_value>0.95 and max_value > 1.20):
        target_value = torch.tensor([1.0])
        weight_value = torch.tensor([10.0])
    
    
    input_tensor = input_tensor.unsqueeze(-1)/torch.mean(input_tensor)
    volume_tensor = volume_tensor.unsqueeze(-1)/torch.mean(volume_tensor)
    
    feature_tensor = torch.concat([input_tensor,volume_tensor],dim = 1)
    return feature_tensor, target_value, weight_value


def feature_target_plot(df, feature_start_index, target_start_index, target_end_index):
    """
    normalize input
    """
    input_tensor_origine = torch.tensor(np.asarray(df.iloc[feature_start_index: target_start_index]['close']), dtype = torch.float32)
    
    output_tensor_orgine = torch.tensor(np.asarray(df.iloc[target_start_index: target_end_index]['close']), dtype = torch.float32)
    
    gain_tensor = output_tensor_orgine/input_tensor_origine[-1]
    
    min_value = torch.min(gain_tensor)
    max_value = torch.max(gain_tensor)
    
    target_value = torch.tensor([0.0])
    weight_value = torch.tensor([1.0])
    if(min_value>0.95 and max_value > 1.20):
        target_value = torch.tensor([1.0])
        weight_value = torch.tensor([10.0])
    
    
    input_tensor = input_tensor_origine/torch.mean(input_tensor_origine)
    return input_tensor, target_value, weight_value, input_tensor_origine, output_tensor_orgine


#%%

class BaostockDailyDataset(Dataset):
    def __init__(self, data_folder, reference_file, transform=None):
        # get the stock list and total length
        self.df_ref = pd.read_csv(reference_file)
        self.data_folder = data_folder
        

    def __len__(self):
        
        return len(self.df_ref)

    def __getitem__(self, index):
        data_file_name = self.df_ref.iloc[index]["code"]
        feature_start_index = self.df_ref.iloc[index]["feature_start_index"]
        target_start_index = self.df_ref.iloc[index]["target_start_index"]
        target_end_index = self.df_ref.iloc[index]["target_end_index"]
        df_data = pd.read_csv(os.path.join(self.data_folder, data_file_name))
        
        feature, target,weight = feature_target_1(df_data, feature_start_index, target_start_index, target_end_index)
        
        
        return feature, target ,weight



class BaostockDailyDatasetTotalRead(Dataset):
    def __init__(self, data_folder, reference_file,data_file,dict_file,transform=None):
        # get the stock list and total length
        self.df_ref = pd.read_csv(reference_file)
        self.data_folder = data_folder
        self.df_all = pd.read_csv(data_file)
        self.index_map  = pickle.load(open(dict_file,'rb'))
        

    def __len__(self):
        
        return len(self.df_ref)

    def __getitem__(self, index):
        data_file_name = self.df_ref.iloc[index]["code"]
        feature_start_index = self.df_ref.iloc[index]["feature_start_index"]
        target_start_index = self.df_ref.iloc[index]["target_start_index"]
        target_end_index = self.df_ref.iloc[index]["target_end_index"]
        
        start_index, end_index = self.index_map[data_file_name]
        # print(start_index, end_index)
        
        df_data = self.df_all.loc[start_index: end_index]
        # print("yes")
        df_data = df_data.reset_index(drop = True)
        
        feature, target,weight = feature_target_1(df_data, feature_start_index, target_start_index, target_end_index)
        
        
        return feature, target ,weight


class BaostockDailyDatasetPlot(Dataset):
    def __init__(self, data_folder, reference_file, transform=None):
        # get the stock list and total length
        self.df_ref = pd.read_csv(reference_file)
        self.data_folder = data_folder
        

    def __len__(self):
        
        return len(self.df_ref)

    def __getitem__(self, index):
        data_file_name = self.df_ref.iloc[index]["code"]
        feature_start_index = self.df_ref.iloc[index]["feature_start_index"]
        target_start_index = self.df_ref.iloc[index]["target_start_index"]
        target_end_index = self.df_ref.iloc[index]["target_end_index"]
        df_data = pd.read_csv(os.path.join(self.data_folder, data_file_name))
        
        feature, target,weight,input_origine, output_origine = feature_target_plot(df_data, feature_start_index, target_start_index, target_end_index)
        
        
        return feature, target ,weight,input_origine, output_origine
       


#%%

def test():
    data_folder = "/home/yao/project/dawin/data/baostock_daily/hasvolume_data/"
    train_reference_file = "/home/yao/project/dawin/code/baostock_dataprepare/baostock_list_train.csv"

    dataset_plot = BaostockDailyDatasetPlot(data_folder,train_reference_file)
    for sample_id,(data, target,weight,input_origine,output_origine) in enumerate(dataset_plot):

        fig = plt.figure(tight_layout = True, figsize = (7,5),)
        ax = fig.add_subplot()
        ax.plot(np.arange(len(input_origine)),input_origine)
        ax.plot(np.arange(len(input_origine), len(input_origine) + len(output_origine)),output_origine)
        
        plt.savefig("./plot_figure/plot_" + str(sample_id).zfill(5) + ".png",dpi = 500, bbox_inches = "tight")
        plt.close(fig)
        plt.clf()
        plt.cla()

if __name__ == "__main__":
    test()

    

    
    
