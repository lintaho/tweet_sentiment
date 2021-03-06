# Streams stock data from yahoo finance API by minute

import ystockquote
import time
from datetime import datetime
from pymongo import MongoClient

# MongoDB Setup
connection = MongoClient('localhost', 27017)
db = connection['trading_day_' + datetime.now().strftime('%U%w')]

def load_stocklist(document):
    stock_list = []
    f = open(document)
    for symb in f.readlines():
        stock_list.append(symb.strip())
    return stock_list

stock_list = load_stocklist('tech_etfs.txt')

# Queries Yahoo API for list of ETF prices every minute.
while True:
    now = datetime.now().strftime('%H%M')
    day = datetime.now().weekday()
    if '0830' <= now <= '1500' and day < 5:
        try:
            for stock in stock_list:
                col = db[stock]
                price = ystockquote.get_price(stock)
                print "Current price of " + stock + ": " + price
                col.insert({"quote": stock, "price": price, "time": datetime.now().strftime('%H%M%S'), "day": day, "week": datetime.now().strftime('%U')})
            time.sleep(60)
        except Exception, e:
            continue
