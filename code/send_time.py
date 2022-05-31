#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 22:55:43 2022

@author: eason
"""
# 接口来源 
import akshare as ak
import datetime 
from tools.message_tools import Send_Message
import time



stock_list = ["002468", "002241","603918", 
              "000404", "002351","000909", 
              "603628"]
mailed_map = {}

current_time = datetime.datetime.now()
# 可以改变时间，用于测试用
# current_time  = datetime.datetime(2022, 5, 29, 0, 0, 0)


send_message = Send_Message()

while(1):
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
               (news_public_datetime.date() == current_time.date() ) ):
                # 发送邮件
                print("发送邮件")
                send_message.send_to_myself(subject = stock_code + ":" + news_public_time + ":" + news_title  ,
                                            content = ",".join([stock_code, news_title, news_content, news_public_time,  news_url]))
                mailed_map[news_url] = 1
    # 间隔5分钟发送
    time.sleep(600)