

#%%
import baostock as bs
import pandas as pd
from .download_tools import *

def get_stocklist(stock_type = "hs300"):
    """
    返回沪深300的指标股 
    """
    # 登陆系统
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:'+lg.error_code)
    print('login respond  error_msg:'+lg.error_msg)

    # 获取沪深300成分股
    if(stock_type == "hs300"):
        rs = bs.query_hs300_stocks()
    elif(stock_type == "zz500"):
        rs = bs.query_zz500_stocks()
    elif(stock_type == "sz50"):
        rs = bs.query_sz50_stocks()

    print(f'query_{stock_type} error_code:'+rs.error_code)
    print(f'query_{stock_type}  error_msg:'+rs.error_msg)

    # 打印结果集
    stocks_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        stocks_list.append(rs.get_row_data())
    result = pd.DataFrame(stocks_list, columns=rs.fields)
    
    # 结果集输出到csv文件
    # result.to_csv("D:/hs300_stocks.csv", encoding="gbk", index=False)
    # print(result)

    # 登出系统
    bs.logout()
    return list(result["code"])




def test():
    result = get_stocklist()
    print(result)

test()
    
# %%
