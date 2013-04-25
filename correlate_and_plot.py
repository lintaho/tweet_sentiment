# Plots a set of positive/negative tweets onto a time series based on their ratio

from pymongo import MongoClient
import matplotlib.pyplot as plt
from numpy import *
from scipy import stats as sp

# MongoDB setup
connection = MongoClient('localhost', 27017)
db_stocks = connection.trading_day_123
db_tweets = connection.data
results_col_stocks = db_stocks['XLK']
results_col_tweets = db_tweets['results123']

# Market open and close times
begin_time = 83000
end_time = 150000

# Converts all prices to a list indexed by timestamp
all_prices = []
lt = '083000'
for minute in range(420):
    gt = "%02d" % (minute / 60 + 8) + ("%02d" % ((minute) % 60)) + '00'
    if gt > lt:
        s = results_col_stocks.find({'time': {'$gt': lt, '$lt': gt}})
        if s.count() != 0:
            for x in range(1):
                all_prices.append(float(s[x]['price']))
        else:
            if len(all_prices) > 1:
                all_prices.append(float(all_prices[len(all_prices) - 1]))
            else:
                all_prices.append(0.0)
        lt = gt

prices = []
times = []
for i in range(s.count()):
    prices.append(float(s[i]['price']))
    times.append(s[i]['time'])

# Adds tweets to list indexed by timestamp
lt = '083000'
all_tweet_scores = []
for minute in range(420):
    gt = "%02d" % (minute / 60 + 8) + ("%02d" % ((minute) % 60)) + '00'
    print gt
    if gt > lt:
        b = results_col_tweets.find({'time': {'$gt': lt, '$lt': gt}})
        if b.count() != 0:  # if one or more tweets in that minute
            p, n = 0, 0
            for y in range(int(b.count() * .5)):  # for each tweet, add up the pos and neg scores
                sent = b[y]['sent']
                if sent == 'positive':
                    p += 1
                else:
                    n += 1
            if (p + n) != 0:
                r = float(p) / float(p + n)
            else:
                r = float(p)
            all_tweet_scores.append(r)
        else:  # if non tweets in that minute
            if len(all_tweet_scores) > 1:
                all_tweet_scores.append(all_tweet_scores[len(all_tweet_scores) - 1])
            else:
                all_tweet_scores.append(0.0)
        lt = gt

lags = [10, 20, 30, 40, 50, 60, 70, 80, 90]  # various lags we will test minutes
for lag in lags:
    lagged_prices = all_prices[0:len(all_prices) - lag]
    fixed_tweets = all_tweet_scores[lag:]
    print str(sp.stats.pearsonr(lagged_prices, fixed_tweets)) + 'for lag: ' + str(lag)

b = array(all_tweet_scores)
b /= b.max()
a = array(all_prices)
a /= a.max()

plt.plot(a)
plt.axis([0, len(all_tweet_scores), 0, 1])
plt.plot(b)
plt.show()
