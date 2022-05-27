#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 21:19:33 2021

@author: yao
"""
"""
https://cloud.tencent.com/developer/article/1702300
"""

import smtplib  # 导入PyEmail
from email.mime.text import MIMEText
import time
# import schedule
import datetime

class Send_Message():

    def __init__(self):
        self.sender = "yaoyichen23@163.com"  # 发送方
        self.recver = "yaoyichen23@163.com"  # 接收方
        self.password = "kimi0923" #邮箱密码

    def send_to_myself(self, subject,content):
        subject = subject  # 邮件标题
        content = content
        
        message = MIMEText(content, "plain", "utf-8")
        # content 发送内容     "plain"文本格式   utf-8 编码格式

        message['Subject'] = subject  # 邮件标题
        message['To'] = self.recver  # 收件人
        message['From'] = self.sender  # 发件人

        smtp = smtplib.SMTP_SSL("smtp.163.com", 994)  # 实例化smtp服务器
        smtp.login(self.sender, self.password)  # 发件人登录
        smtp.sendmail(self.sender, [self.recver], message.as_string())  # as_string 对 message 的消息进行了封装
        smtp.close()
        print("发送邮件成功！！")



def sample_sendemail():
    """
    发送邮件给自己,可以指定主题和内容
    """
    send_message = Send_Message()

    time.sleep(1)

    current_time = str(datetime.datetime.now())
    print(type(current_time))
    print(current_time)
    send_message.send_to_myself(subject = "a",content =  current_time)
    return 0


if __name__ == "__main__":
    sample_sendemail()