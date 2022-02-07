#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 22:44:24 2021

@author: yao
"""
import backtrader as bt
import datetime
import pandas as pd

import torch

net = torch.nn.Embedding(10000,128)
for parameter in net.parameters():
    print(parameter.shape, parameter.shape.numel())