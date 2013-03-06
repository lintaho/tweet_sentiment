# Plots a set of positive/negative tweets onto a time series based on their ratio

from pymongo import MongoClient
import matplotlib.pyplot as plt
from numpy import *

connection = MongoClient('localhost', 27017)
db_stocks = connection.trading_day_082
db_tweets = connection.data
# results_col = db[str(sys.argv[1])]

results_col_stocks = db_stocks['SOXL']
results_col_tweets = db_tweets['results']


begin_time = 83000
end_time = 150000



# 390 minutes
all_prices = []
lt = '083000'
for minute in range(420):
    gt = "%02d" % (minute / 60 + 8) + ("%02d" % ((minute) % 60)) + '00'
    if gt > lt:
        # print gt, lt
        s = results_col_stocks.find({'time': {'$gt': lt, '$lt': gt}})
        if s.count() != 0:
            for x in range(s.count()):
                all_prices.append(s[x]['price'])
        else:
            if len(all_prices) > 1:
                all_prices.append(all_prices[len(all_prices) - 1])
            else:
                all_prices.append(0)

        t = results_col_tweets.find({'time': {'$gt': lt, '$lt': gt}})


        lt = gt

print len(all_prices)
prices = []
times = []
for i in range(s.count()):
    prices.append(float(s[i]['price']))
    times.append(s[i]['time'])
lt = '083000'
all_tweet_scores = []
for minute in range(420):
    gt = "%02d" % (minute / 60 + 8) + ("%02d" % ((minute) % 60)) + '00'
    print gt
    if gt > lt:
        b = results_col_tweets.find({'time': {'$gt': lt, '$lt': gt}})
        if b.count() != 0:  # if one or more tweets in that minute
            p, n = 0, 0
            for y in range(b.count()):  # for each tweet, add up the pos and neg scores
                sent = b[y]['sent']
                if sent == 'positive':
                    p += 1
                else:
                    n += 1
            if n != 0:
                r = float(p) / float(n)
            else:
                r = 1.0
            all_tweet_scores.append(r)
        else:  # if non tweets in that minute
            if len(all_tweet_scores) > 1:
                all_tweet_scores.append(all_tweet_scores[len(all_tweet_scores) - 1])
            else:
                all_tweet_scores.append(0)
print all_prices, all_tweet_scores
print len(all_prices), len(all_tweet_scores)
# scores = []
# t = results_col_stocks.find()
# time = t[0]['time']
# pos_score, neg_score = 0, 0
# for tweet_index in range(t.count()):
#     sent = t[tweet_index]['sent']
#     new_time = t[tweet_index]['time']
#     if new_time == time:
#         if sent == 'positive':
#             pos_score += 1
#         else:
#             neg_score += 1
#     else:
#         if neg_score != 0:
#             ratio = float(pos_score) / float(neg_score)
#         else:
#             ratio = 1.0
#         scores.append(ratio)
#         pos_score, neg_score = 0, 0
#         if sent == 'positive':
#             pos_score += 1
#         else:
#             neg_score += 1
#         time = new_time
 
b = array(all_tweet_scores)
b /= b.max()
a = array(all_prices)
a /= a.max()

plt.plot(a)
plt.axis([0, len(scores), 0, 1])
plt.plot(b)
plt.show()
