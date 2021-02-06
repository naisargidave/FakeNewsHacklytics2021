# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 21:22:41 2021

@author: naisa
"""

article = "Cable News Network is a multinational news-based pay television channel headquartered in Atlanta. It is owned by CNN Worldwide, a unit of the WarnerMedia News & Sports division of AT&T's WarnerMedia. It was founded in 1980 by American media proprietor Ted Turner and Reese Schonfeld as a 24-hour cable news channel"    
    
import requests
url = 'http://127.0.0.1:8080/predict'
myobj = {'article': article}

x = requests.post(url, json = myobj)
print(x.status_code)
print(x)