#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 23:06:08 2021

@author: yao
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import sys
import datetime
import backtrader as bt



import matplotlib
import matplotlib.pyplot as plt
from tools.download_data import *

matplotlib.use('Qt5Agg')
plt.switch_backend('Qt5Agg')





start_datestr, end_datestr = "2020-01-01", "2021-01-01"
stock_code = "600000"
stock_market = "sh"  # "sh,sz"
result3 = get_daily_from_baostock(stock_code,stock_market,start_datestr, end_datestr)
result3 = result3.rename(columns={"date":"datetime"})
result3 = result3.rename(columns = {"datetime":"Date",
                "open":"Open",
                "high":"High",
                "low":"Low",
                "close":"Close",
                "volume":"Volume"})
result3["Adj Close"] = result3["Close"]
result3 = result3[["Date","Open","High","Low","Close","Adj Close","Volume"]]
result3.set_index("Date", inplace= True)
result3.to_csv("../data/600000.csv", index=True)


#%%
# Create a Stratey
class TestStrategy(bt.Strategy):

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, {:.2f}, {:.2f}'.format(self.dataclose[0],  self.dataclose[-1]))

        if self.dataclose[0] < self.dataclose[-1]:
            # current close less than previous close

            if self.dataclose[-1] < self.dataclose[-2]:
                # previous close less than the previous close

                # BUY, BUY, BUY!!! (with all possible default parameters)
                self.log('BUY CREATE, %.2f' % self.dataclose[0])
                self.buy()
        

cerebro = bt.Cerebro()


# Datas are in a subfolder of the samples. Need to find where the script is
# because it could have been called from anywhere
modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
datapath = os.path.join(modpath, "../data/600000.csv")

# Create a Data Feed
data = bt.feeds.YahooFinanceCSVData(
    dataname=datapath,
    # Do not pass values before this date
    fromdate=datetime.datetime(2020, 1, 1),
    # Do not pass values after this date
    todate=datetime.datetime(2021 ,1, 1),
    reverse=False)



    # Create a Data Feed
# data = bt.feeds.YahooFinanceCSVData(
#     dataname="../data/orcl-1995-2014.txt",
#     # Do not pass values before this date
#     fromdate=datetime.datetime(2000, 1, 1),
#     # Do not pass values after this date
#     todate=datetime.datetime(2000, 12, 31),
#     reverse=False)
# Add the Data Feed to Cerebro
cerebro.adddata(data)

# Add a strategy
cerebro.addstrategy(TestStrategy)

cerebro.broker.setcash(100000.0)

print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

cerebro.run()
# cerebro.plot(height= 30,iplot=False)
# cerebro.plot(iplot=False,volume=False, savefig=True, figfilename='backtrader-plot.png')


figure = cerebro.plot(height= 50,style = 'candlebars',iplot=False)[0][0]
figure.savefig('example.png')


print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())