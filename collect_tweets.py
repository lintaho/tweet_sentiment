from pymongo import MongoClient
from twython import Twython
import time
import datetime
connection = MongoClient('localhost', 27017)
db = connection.local
collection = db['tweet_series']

t = Twython(app_key='tukYVvz2d4MIG4KCxObMkA',
                    app_secret='uxJwgj45Qi1lbuqz6FuYcZJsTvpKG6fGab46BBsZ28',
                                                oauth_token='49249889-yILZ38P5XB84kCV0ZvBY2X4iQg0FdrBxE02mAParR',
                                                                                        oauth_token_secret='hVKkn8jlIyUQbKPnWXnlmdL4j8iM7FDI7LFOtzX40')

sid = None
for db_num in range(11): #start at an hour
  collection = db['marley_tweets' + str(db_num) ]
  for x in range(327):
    print "Time remaining: " + str(datetime.timedelta(seconds=(327-x)*11)) + " on hour " + str(db_num)
    try:
      tweets=t.search(q='Bob Marley OR #OneLove OR marley #bobmarley', count='100', lang='en', since_id=sid, result_type='recent')
          #stock OR stocks OR market OR finance OR exchange OR economy', count='100', lang='en', since_id=sid, result_type='recent')
    except:
      time.sleep(110)
      x+=10
      continue 
    for i in range(len(tweets['statuses'])):
      collection.insert({"tweet_id": tweets['statuses'][i]['id_str'],
                         "text": tweets['statuses'][i]['text'], 
                         "user_id": tweets['statuses'][i]['user']['id_str']})
    if len(tweets) > 0:
      sid = tweets['search_metadata']['max_id_str']
    time.sleep(11)
  time.sleep(3)

