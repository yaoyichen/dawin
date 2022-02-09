#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 23:38:03 2022

@author: yao
"""

import pandas as pd
import os

# shift span of the data


data_dir = "/home/yao/project/dawin/data/baostock_daily/hasvolume_data"
shift_span = 20

file_list = os.listdir(data_dir)

for file in file_list:
    print(file)