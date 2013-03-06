from pymongo import MongoClient
import math

connection = MongoClient('localhost', 27017)
db_stocks = connection.trading_day_092
db_tweets = connection.data

results_col_stocks = db_stocks['FDN']
results_col_tweets = db_tweets['tech092']

sw_file = open("stopwords.txt", "r")
stop_words = []
line = sw_file.readline()
while line:
    words = line.strip()
    stop_words.append(words)
    line = sw_file.readline()
sw_file.close()



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

# print all_prices

lt = '083000'
all_tweet_words = []
for minute in range(420):
    gt = "%02d" % (minute / 60 + 8) + ("%02d" % ((minute) % 60)) + '00'
    print gt
    if gt > lt:
        b = results_col_tweets.find({'time': {'$gt': lt, '$lt': gt}})
        if b.count() != 0:  # if one or more tweets in that minute
            words = []
            for y in range(b.count()):
                txt = b[y]['text'].split(' ')
                for w in txt:
                    if w in stop_words:
                        continue
                    else:
                        words.append(w)
            all_tweet_words.append(words)
        else:  # if non tweets in that minute
            if len(all_tweet_words) > 1:
                all_tweet_words.append(all_tweet_words[len(all_tweet_words) - 1])
            else:
                all_tweet_words.append([])
    # print all_tweet_words
# print all_prices, all_tweet_words
print len(all_prices), len(all_tweet_words)

lag = 10  # minutes

word_scores = {}
for minute in range(390 - lag):
    twords = all_tweet_words[minute]
    print twords
    price = all_prices[minute + lag - 1]
    div = float(price) / float(all_prices[minute])
    for word in twords:
        if word in word_scores:
            word_scores[word] += math.log(div)
        else:
            word_scores[word] = 0

print len(word_scores)
for w in sorted(word_scores, key=word_scores.get, reverse=False):
    print w, word_scores[w]
