import os
os.sys.path.append("..")

from typing import OrderedDict
import backtrader as bt
from datetime import datetime

cerebro  = bt.Cerebro()
cerebro.getbroker().set_cash(100000)

# 
perc = 0.0000
cerebro.broker.set_slippage_perc(perc = perc)
cerebro.broker.setcommission(0.000)


data = bt.feeds.YahooFinanceCSVData(dataname = "MSFT",fromdate= datetime(2020,3,3),todate= datetime(2021,3,3))

cerebro.adddata(data)

print(data)

cerebro.run()