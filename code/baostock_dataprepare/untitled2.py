#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 22:31:19 2022

@author: eason
"""
# 接口来源 
import akshare as ak
import datetime 

stock_list = ["002468", "002241"]
mailed_map = {}

current_time = datetime.datetime.now()
current_time  = datetime.datetime(2022, 5, 16, 0, 0, 0)


for stock_code in stock_list:
    stock_news_em_df = ak.stock_news_em(stock = stock_code)
    for index,stock_news_value in stock_news_em_df.iterrows():
        news_code = stock_news_value.code 
        news_title = stock_news_value.title
        news_content = stock_news_value.content
        news_public_time = stock_news_value.public_time
        news_url = stock_news_value.url
        
        # print(news_title)
        news_public_datetime = datetime.datetime.strptime(news_public_time, "%Y-%m-%d %H:%M:%S")
        # 如果当天的新闻，且未被发布过
        if((news_url not in mailed_map ) and
           ((news_public_datetime - current_time).days ==0 ) ):
            # 发送邮件
            print("发送邮件")
            
            mailed_map[news_url] = 1

            