# Streams tweets from twitter throughout a day, to correlate with stock data time periods

from pymongo import MongoClient
from twython import Twython
import time
from datetime import datetime

# MongoDB Setup
connection = MongoClient('localhost', 27017)
db = connection.data

# Twitter API Info
t = Twython(app_key='tukYVvz2d4MIG4KCxObMkA',
            app_secret='uxJwgj45Qi1lbuqz6FuYcZJsTvpKG6fGab46BBsZ28',
            oauth_token='49249889-yILZ38P5XB84kCV0ZvBY2X4iQg0FdrBxE02mAParR',
            oauth_token_secret='hVKkn8jlIyUQbKPnWXnlmdL4j8iM7FDI7LFOtzX40'
            )

sid = None
query = 'technology OR stocks'

# Contiously searches tweets for query at max rate limit
while True:
    now = datetime.now().strftime('%H%M')
    day = datetime.now().weekday()
    if '0830' <= now <= '1500' and day < 5:
        collection = db['tech' + datetime.now().strftime('%U%w')]
        for x in range(327):
            # print 'Time remaining: ' + str(datetime.timedelta(seconds=(327 - x) * 11))
            try:
                tweets = \
                    t.search(q=query,
                             count='100', lang='en', since_id=sid,
                             result_type='recent')
            except:
                time.sleep(110)
                print 'rate limited'
                x += 10
                continue
            for i in range(len(tweets['statuses'])):
                collection.insert({'tweet_id': tweets['statuses'][i]['id_str'],
                                    'text': tweets['statuses'][i]['text'],
                                   'user_id': tweets['statuses'][i]['user']['id_str'],
                                   "time": datetime.now().strftime('%H%M%S'),
                                   "day": day,
                                   "week": datetime.now().strftime('%U')})
            print 'Added ' + str(len(tweets['statuses'])) + ' tweets'
            if len(tweets) > 0:
                sid = tweets['search_metadata']['max_id_str']
            time.sleep(11)
