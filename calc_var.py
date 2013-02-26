# Calculates the Realized Variance of a set of data

import numpy as np
from pymongo import MongoClient

connection = MongoClient('localhost', 27017)


def load_stocklist(document):
    stock_list = []
    f = open(document)
    for symb in f.readlines():
        stock_list.append(symb.strip())
    return stock_list

week = '08'
days = ['1', '2', '3', '4', '5']
stock_list = load_stocklist('tech_etfs.txt')

def calc_RVs(week, day):   # str str
    db = connection['trading_day_' + week + day]
    for stock in stock_list:
        col = db[stock]
        for entry in col.find():
            print entry

   # print np.absolute(-10)


calc_RVs('07', '6')
