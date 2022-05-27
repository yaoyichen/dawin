#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 22:25:44 2022

@author: yao
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 21:02:42 2021

@author: yao
"""
import os
os.sys.path.append("..")
from tools.download_tools import *
from tools.get_product_code import get_hs300_list
from tools.common_tools import  timing,check_valid_date


start_datestr, end_datestr = "2015-01-01", "2022-05-29"
stock_code = "600000"
stock_market = "sh"  # "sh,sz"

hs300_stock_list = get_hs300_list()

for item in hs300_stock_list:
    stock_market,stock_code = item.split(".")

    result = get_daily_from_baostock(stock_code,stock_market,start_datestr, end_datestr)
    data_dir = "../../data/baostock_daily"
    result.to_csv(os.path.join(data_dir,f"stock_{stock_code}.csv"),index = False)