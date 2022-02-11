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



#%%


def feature_target_1(df, feature_start_index, target_start_index, target_end_index):
    """
    normalize input
    
    """
    input_tensor = torch.tensor(np.asarray(df.iloc[feature_start_index: target_start_index]['close']), dtype = torch.float32)
    
    output_tensor = torch.tensor(np.asarray(df.iloc[target_start_index: target_end_index]['close']), dtype = torch.float32)
    
    gain_tensor = output_tensor/input_tensor[-1]
    
    min_value = torch.min(gain_tensor)
    max_value = torch.max(gain_tensor)
    
    target_value = torch.tensor([0.0])
    weight_value = torch.tensor([1.0])
    if(min_value>0.95 and max_value > 1.30):
        target_value = torch.tensor([1.0])
        weight_value = torch.tensor([10.0])
    
    
    input_tensor = input_tensor/torch.mean(input_tensor)
    return input_tensor, target_value, weight_value


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
        
        
train_reference_file = "/home/yao/project/dawin/code/baostock_dataprepare/baostock_list_train.csv"
test_reference_file = "/home/yao/project/dawin/code/baostock_dataprepare/baostock_list_test.csv"

data_folder = "/home/yao/project/dawin/data/baostock_daily/hasvolume_data/"


batch_size = 200
train_set = BaostockDailyDataset(data_folder,train_reference_file)
test_set = BaostockDailyDataset(data_folder,test_reference_file)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)


class RNN_LSTM(nn.Module):
    def __init__(self, feature_channnel, hidden_size, num_layers):
        super(RNN_LSTM, self).__init__()
        self.feature_channnel = feature_channnel
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(feature_channnel, hidden_size, num_layers, batch_first = True,bidirectional = True)
        self.conv1d = nn.Conv1d(2*hidden_size, 1, kernel_size = 1, padding=0 ,bias = True)
        
        
    def forward(self,x):
        
        h0 = torch.zeros(2*self.num_layers, x.shape[0], self.hidden_size, requires_grad= False).to(x.device)
        c0 = torch.zeros(2*self.num_layers, x.shape[0], self.hidden_size, requires_grad= False).to(x.device)
       
        out,(a,b)  = self.lstm(x, (h0,c0))
        out = torch.permute(out[:,-1::,:], [0,2,1])
        out = self.conv1d(out).squeeze(-1)
        out = torch.sigmoid(out)
        
        return out
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNN_LSTM(feature_channnel = 1,hidden_size = 2, num_layers = 3)
model.to(device)
loss_fun = torch.nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr = 0.003)
epoch_number =  100

def evaluate(model, data_loader,loss_fun):
    loss_sum = 0.0
    number_sum = 0
    for batch_idx, (data, target,weight) in enumerate(data_loader):
        data = data.unsqueeze(-1).to(device)
        target = target.to(device)
        pred = model(data)
        loss = loss_fun(pred, target)
    
        loss_sum += loss.item()*len(data)
        number_sum += len(data)
    return loss_sum/ number_sum
        
        
    
for epoch in range(epoch_number):
    for batch_idx, (data, target,weight) in enumerate(train_loader):
        data = data.unsqueeze(-1).to(device)
        target = target.to(device)
        weight = weight.to(device)
        
        pred = model(data)
        
        loss = loss_fun(pred, target)
        # loss = torch.mean(torch.abs(pred - target))
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()
        # print(loss.item())
    if(epoch % 5 ==0):
        train_result = evaluate(model, train_loader,loss_fun)
        test_result = evaluate(model, test_loader,loss_fun)
        print(train_result, test_result)


