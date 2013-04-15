# Script used to mine positive and negative emotion tweets

from twython import Twython
import time
from pymongo import MongoClient
import datetime

# MongoDB setup
connection = MongoClient('localhost', 27017)
db = connection.local
neg_col = db['neg_emoticons']
pos_col = db['pos_emoticons']

t = Twython(app_key='APPKEY',
            app_secret='APPSECRET',
            oauth_token='OAUTHTOKEN',
            oauth_token_secret='OAUTHTOKENSECRET'
            )

num_hours = 3  # pos and neg each
pos_query = ':) OR :-) OR =) OR :D OR :P OR <3 OR like OR love'
neg_query = ":( OR =( OR hate OR dislike"
sid = None

pos, neg = False, True

# Positive emotion tweets
if pos:
    for hour in range(num_hours):
        for x in range(327):  # 1 hour ~= 327
            print 'Time remaining for positive emotion collection: ' + \
                str(datetime.timedelta(seconds=(327 - x) * 11 + 3)) + ' with ' + str(2 * (num_hours - hour)) + ' hours remaining.'
            try:
                tweets = t.search(q=pos_query,
                                  count='100', lang='en', since_id=sid,
                                  result_type='recent')
            except:
                continue
            for i in range(len(tweets['statuses'])):
                pos_col.insert({'text': tweets['statuses'][i]['text'], 'label': 'positive'})
            if len(tweets) > 0:
                sid = tweets['search_metadata']['max_id_str']
            time.sleep(11)
        time.sleep(3)  # rounds up the hour

# Negative emotion tweets
if neg:
    for hour in range(num_hours):
        for y in range(327):
            print 'Time remaining for negative emotion collection: ' + \
                str(datetime.timedelta(seconds=(327 - y) * 11 + 3)) + ' with ' + str(num_hours - hour) + ' hours remaining.'
            try:
                tweets = t.search(q=neg_query,
                                  count='100', lang='en', since_id=sid,
                                  result_type='recent')
            except:
                continue
            for i in range(len(tweets['statuses'])):
                neg_col.insert({'text': tweets['statuses'][i]['text'], 'label': 'negative'})
            if len(tweets) > 0:
                sid = tweets['search_metadata']['max_id_str']
            time.sleep(11)
        time.sleep(3)
