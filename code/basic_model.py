#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 21:02:42 2021

@author: yao
"""

import numpy as np
import torch
import torch.nn as nn
import time
from tools.download_data import *



download = False
if(download):
    df = get_daily_from_baostock("000300", "sh", "2020-01-01", "2021-09-30")
    df.to_csv("../data/sh000300_hist.csv")
else:
    df = pd.read_csv("../data/sh000300_hist.csv")


price_vector = np.asarray(df["close"])
volume_vector = np.asarray(df["volume"])