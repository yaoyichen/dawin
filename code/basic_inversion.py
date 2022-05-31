#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 21:02:42 2021

@author: yao
"""
#%%
from operator import truediv
import numpy as np
import torch
import torch.nn as nn
import time
from tools.download_tools import *
import matplotlib.pyplot as plt
"""
target on sh000300, find an easy model to predict
"""

class Args:
    adjoint = False

args = Args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

#%%
df = pd.read_csv("../data/baostock_daily/stock_399006.csv")


price_vector = np.asarray(df["close"])
volume_vector = np.asarray(df["volume"])

true_price_vector = torch.tensor(price_vector[0:100], dtype = torch.float32)



# %%


# %%
class PriceVolumeModel(nn.Module):
    def __init__(self):
        pass

    def forward(self,t, state):
        velocity, price, force = state

        d_velocity = force
        d_price = velocity
        d_force  = torch.zeros(force.shape, dtype = torch.float32)
        result = torch.stack([d_velocity, d_price, d_force])
        return result





# %%

dt = 1.0 # 1天的间隔
time_vector = torch.tensor(np.arange(0,100,1))


force = torch.randn([100,1])*0.0
force.requires_grad = True
velocity = 10*torch.ones([100,1] ,dtype = torch.float32)
price = torch.ones([100,1] ,dtype = torch.float32)*1451


optimizer = torch.optim.Adam([force], lr = 0.5)
criterion = torch.nn.MSELoss()
model = PriceVolumeModel()

#%%
start_index= 0
for iteration in range(1):
    
    state_all = torch.concat([velocity, price, force], dim = 1).unsqueeze(-1)
    total_result = torch.zeros(state_all.shape, dtype = torch.float32)
    
    state = state_all[0,:,:]
    total_result[0,:,:] = state_all[0,:,:]
    
    for (index, t_) in enumerate(time_vector[0:-1]):
        state[2,:] = state_all[index,2,:]
        
        state = state + model.forward(dt, state) * dt       
        total_result[index+1,:,:] = state
        
    
    # 需要定义一下观测的长度 
    loss = criterion(total_result[start_index: start_index + 20,1,:],
                         true_price_vector[start_index: start_index + 20])
    
    print(loss)
    if(iteration%100 == 0):
        start_index += 1
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
#%%

plt.plot(time_vector[0:50], total_result[0:50 ,1,0].detach().numpy())
plt.plot(time_vector[0:50], true_price_vector[0:50])


# 
# %%
plt.plot(time_vector[0:50], total_result[0:50,2,0].detach().numpy())
# 



#%%
true_price_vector = torch.tensor(price_vector, dtype = torch.float32)
true_volume_vector = torch.tensor(volume_vector, dtype = torch.float32)/1e7

price_fft = torch.fft.rfft(true_price_vector)
volume_fft = torch.fft.rfft(true_volume_vector)



keep_modes = 18
price_fft[keep_modes::] = 0.0
volume_fft[keep_modes::] = 0.0




# price_fft[0:17] = 0.0
# volume_fft[0:17] = 0.0

# price_fft[27::] = 0.0
# volume_fft[27::] = 0.0



plt.figure(0)
plt.loglog(torch.abs(price_fft), "x") 
plt.loglog(torch.abs(volume_fft), "x") 

price_ifft = torch.fft.irfft(price_fft)
volume_ifft = torch.fft.irfft(volume_fft)

plt.figure(1)
plt.plot(true_price_vector)
plt.plot(price_ifft)


plt.figure(2)
plt.plot(true_volume_vector)
plt.plot(volume_ifft)

plt.figure(3)
plt.plot(price_ifft)
plt.plot(volume_ifft)

