#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 21:19:33 2021

@author: yao
"""
import smtplib  # 导入PyEmail
from email.mime.text import MIMEText
import time


# 邮件构建
"""
https://cloud.tencent.com/developer/article/1702300
"""
def send_to_myself(subject,content):
    subject = subject  # 邮件标题
    content = content
    sender = "yaoyichen23@163.com"  # 发送方
    recver = "yaoyichen23@163.com"  # 接收方
    password = "kimi0923" #邮箱密码
    message = MIMEText(content, "plain", "utf-8")
    # content 发送内容     "plain"文本格式   utf-8 编码格式

    message['Subject'] = subject  # 邮件标题
    message['To'] = recver  # 收件人
    message['From'] = sender  # 发件人

    smtp = smtplib.SMTP_SSL("smtp.163.com", 994)  # 实例化smtp服务器
    smtp.login(sender, password)  # 发件人登录
    smtp.sendmail(sender, [recver], message.as_string())  # as_string 对 message 的消息进行了封装
    smtp.close()
    print("发送邮件成功！！")



if __name__=='__main__':
    code = "600000"
    price = 5.0
    subject = "get new suggestion"
    content = f"code:{code}, price:{price}"
    send_to_myself(subject,content)