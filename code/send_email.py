#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 21:19:33 2021

@author: yao
"""
# import smtplib  # 导入PyEmail
# from email.mime.text import MIMEText
import time
import schedule
import datetime
from tools.message_tools import Send_Message


if __name__=='__main__':
    code = "600000"
    price = 5.0
    subject = "get new suggestion"
    content = f"code:{code}, price:{price}"
    # send_to_myself(subject,content)


    # The job will be executed on November 6th, 2009
    #xec_date = date(2009, 11, 6)

    # Store the job in a variable in case we want to cancel it
    # job = sched.add_date_job(my_job, exec_date, ['text'])

    # The job will be executed on November 6th, 2009 at 16:30:05

    send_message = Send_Message()
    while True:
        schedule.run_pending()
        time.sleep(10)

        current_time = str(datetime.datetime.now())
        print(type(current_time))
        print(current_time)
        schedule.every(10).seconds.do(send_message.send_to_myself,"a",current_time)