#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 21:02:42 2021

@author: yao
"""


from functools import wraps
from time import time
import datetime

def timing(f):
    """
    https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator
    """
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return wrap



def datestr2int(datestr):
    """
    return year, month day
    """
    datetime_date = datetime.datetime.strptime(datestr, '%Y-%m-%d')
    return datetime_date.year, datetime_date.month, datetime_date.day



def check_valid_date(date_string):
    """
    date_string = '12-25-2018'
    
    """
    format = '%Y-%m-%d'
    try:
        datetime.datetime.strptime(date_string, format)
        return 1
    except ValueError:
      print("This is the incorrect date string format. It should be YYYY-MM-DD")
      return 0