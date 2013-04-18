# Script used to find the most significant words in a group of tweets. Includes bigrams and trigrams
from __future__ import print_function
from pymongo import MongoClient
import nltk
import re
from nltk.corpus import stopwords


# MongoDB setup
connection = MongoClient('localhost', 27017)
db_stocks = connection.trading_day_092
db_tweets = connection.data
col_names = ['tech121', 'tech122', 'tech123', 'tech124', 'tech125']
results_col_stocks = db_stocks['FDN']
results_col_tweets = db_tweets['tech121']

stop_words = stopwords.words('english')
stop_words.append('rt')
stop_words.append('technology')
stop_words.append('stocks')

# Iterate through prices, and store them as a list.
for col in col_names:
    results_col_tweets = db_tweets[col]
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

    # Iterate through tweets, and store each word, bigram, and trigram, as a list
    lt = '083000'
    all_tweet_words = []
    for minute in range(420):
        gt = "%02d" % (minute / 60 + 8) + ("%02d" % ((minute) % 60)) + '00'
        print(gt)
        if gt > lt:
            b = results_col_tweets.find({'time': {'$gt': lt, '$lt': gt}})
            if b.count() != 0:  # if one or more tweets in that minute
                words = []
                for y in range(int(b.count())):
                    txt = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\S+:\/\/\w+)", '', b[y]['text'].lower()).split()
                    bg = nltk.util.bigrams(txt)
                    tg = nltk.util.trigrams(txt)
                    for w in txt:
                        lw = w.lower()
                        if lw in stop_words:
                            continue
                        else:
                            words.append(lw)
                    words.extend(bg)
                    words.extend(tg)
                all_tweet_words.append(words)
            else:  # if non tweets in that minute
                if len(all_tweet_words) > 1:
                    all_tweet_words.append(all_tweet_words[len(all_tweet_words) - 1])
                else:
                    all_tweet_words.append([])
            lt = gt
    lags = [10, 20, 30, 40, 50, 60, 70, 80, 90]  # various lags we will test minutes
    f = open('best_words' + col + '.txt', 'w')
    for lag in lags:
        word_scores = {}
        for minute in range(390 - lag):
            twords = all_tweet_words[minute]
            # print twords
            price = all_prices[minute + lag - 1]
            div = float(price) - float(all_prices[minute])
            for word in twords:
                if word in word_scores:
                    word_scores[word] += div
                else:
                    word_scores[word] = 0

        # print len(word_scores)
        num = 20
        print('lag:' + str(lag), file=f)
        print('Least-----------------------------------------------------------', file=f)
        for i, w in enumerate(sorted(word_scores, key=word_scores.get, reverse=False)):
            if i < num:
                # print w, word_scores[w]
                print(str(w) + ' ' + str(word_scores[w]), file=f)
            if i == (len(word_scores) - num - 1):
                print('Most-----------------------------------------------------------', file=f)
            if i > (len(word_scores) - num):
                print(str(w) + ' ' + str(word_scores[w]), file=f)
        print('\n', file=f)
    f.close()
