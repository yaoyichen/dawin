

#%%
import baostock as bs
import akshare as ak
import pandas as pd
from .download_tools import *

def get_stocklist(stock_type = "hs300"):

    """
    返回沪深300的指标股 
    stock_type in ("hs300", "zz500","sz50","a","kcb")
    """
    # 登陆系统
    if(stock_type in ["hs300", "zz500","sz50"]):
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
        code_list = list(result["code"])
    elif(stock_type in ["a", "kcb"]):
        def add_stockmarket(stock_code):
            if(stock_code[0] == "6"):
                stock_market = "sh"
            else:
                stock_market = "sz"
            return stock_market + '.' + stock_code

        if(stock_type == "a"):
            """
            A股
            """
            stock_zh_a_spot_em_df = ak.stock_zh_a_spot_em()
            code_list_temp = list(stock_zh_a_spot_em_df['代码'])

        elif(stock_type == "kcb"):
            """
            科创板
            """
            stock_zh_kcb_spot_df = ak.stock_zh_kcb_spot()
            code_list_temp = list(stock_zh_kcb_spot_df['代码'])

            

        code_list = [add_stockmarket(i) for i in code_list_temp]

    
    return code_list




def test():
    result = get_stocklist(stock_type = "a")
    print(result)


if __name__ == "__main__":
    test()
    
# %%
