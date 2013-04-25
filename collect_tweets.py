# Script used to collect tweets about any given query by hour

from pymongo import MongoClient
from twython import Twython
import time
import datetime

# MongoDB setup
connection = MongoClient('localhost', 27017)
db = connection.data
collection = db['unlabeled']

t = Twython(app_key='APPKEY',
            app_secret='APPSECRET',
            oauth_token='OAUTHTOKEN',
            oauth_token_secret='OAUTHTOKENSECRET'
            )

query = 'QUERY'
sid = None
num_hours = 5

for db_num in range(num_hours):
    for x in range(327):
        print 'Time remaining: ' + str(datetime.timedelta(seconds=(327 - x) * 11)) + ' on hour ' + str(db_num)
        try:
            tweets = t.search(q=query,
                         count='100', lang='en', since_id=sid,
                         result_type='recent')
        except:
            time.sleep(11)
            x += 10
            continue
        for i in range(len(tweets['statuses'])):
            collection.insert({'tweet_id': tweets['statuses'][i]['id_str'],
                                'text': tweets['statuses'][i]['text'],
                               'user_id': tweets['statuses'][i]['user']['id_str']})
        if len(tweets) > 0:
            sid = tweets['search_metadata']['max_id_str']
        time.sleep(11)
    time.sleep(3)
