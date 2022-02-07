import backtrader as bt
import datetime

class MAStrategy(bt.Strategy):

    params = (("maperiod",60),)

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.sma = bt.indicators.SimpleMovingAverage(self.datas[0], period = self.params.maperiod)

    
    def notify_order(self, order):
        if(order.status in [order.Submitted, order.Accepted] ):
            return
        
        if(order.status in [order.Completed]):
            if order.isbuy():
                self.log(f"BUY at {order.executed.price}")
            
            if order.issell():
                self.log(f"SELL at {order.executed.price}")
            
            self.bar_executed = len(self)
    
        if(order.status in [order.Canceled, order.Margin, order.Rejected]):
            self.log("ORDER FAIL")
        
        self.order = None


    
    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print(dt.isoformat(),txt)

    
    
    def next(self):

        self.log(f"CLOSE at {self.dataclose[0]}")

        if(self.order):
            return
        
        if(not self.position):
            if(self.dataclose[0] > self.sma[0]):
                self.log(f"BUY CREATE:{self.dataclose[0]}")
                self.buy(price = self.dataclose[0], size = 100)
        
        else:
            if(self.dataclose[0] < self.sma[0]):
                self.log(f"SELL CREATE:{self.dataclose[0]}")
                self.sell(price = self.dataclose[0], size = 100)
