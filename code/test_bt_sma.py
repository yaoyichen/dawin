import os
os.sys.path.append("..")

from typing import OrderedDict
import backtrader as bt
from datetime import datetime
from strategy import ma_strategy

cerebro  = bt.Cerebro()
cerebro.getbroker().set_cash(100000)

# 
perc = 0.0000
cerebro.broker.set_slippage_perc(perc = perc)
cerebro.broker.setcommission(0.000)


data = bt.feeds.YahooFinanceCSVData(dataname = "../data/600000.csv",fromdate= datetime(2020,1,1),todate= datetime(2020,12,31))

cerebro.adddata(data)
cerebro.addstrategy(ma_strategy.MAStrategy)

print(data)

print(f"start portfolio:{cerebro.getbroker().getvalue()}")
cerebro.run()
print(f"end portfolio:{cerebro.getbroker().getvalue()}")