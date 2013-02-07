from twython import Twython
import time
from pymongo import MongoClient
import datetime

connection= MongoClient('localhost', 27017)
db = connection.local
neg_col = db['neg_emoticons']
pos_col = db['pos_emoticons']

t = Twython(app_key='tukYVvz2d4MIG4KCxObMkA',
                app_secret='uxJwgj45Qi1lbuqz6FuYcZJsTvpKG6fGab46BBsZ28',
                            oauth_token='49249889-yILZ38P5XB84kCV0ZvBY2X4iQg0FdrBxE02mAParR',
                                        oauth_token_secret='hVKkn8jlIyUQbKPnWXnlmdL4j8iM7FDI7LFOtzX40')

sid = None
for x in range(387):
  print "Time remaining: " + str(datetime.timedelta(seconds=(387-x)*11))
  try:
    tweets=t.search(q=':) OR :-) OR =) OR :D OR :P OR <3 OR <33 OR <333', count='100', lang='en', since_id=sid, result_type='recent')
  except:
    continue
  for i in range(len(tweets['statuses'])):

    #tokenize text here
    pos_col.insert({"text": tweets['statuses'][i]['text'], "label":"positive"})

  if len(tweets) > 0:
    sid = tweets['search_metadata']['max_id_str']

  time.sleep(11)

for y in range(1500):
  print "Time remaining: " + str(datetime.timedelta(seconds=(1500-y)*11))
  try:
    tweets=t.search(q=':( OR :-( OR D: OR ;( OR =( OR T_T OR :\'(', count='100', lang='en', since_id=sid, result_type='recent')
  except:
    continue
  for i in range(len(tweets['statuses'])):

    #tokenize text here
    neg_col.insert({"text": tweets['statuses'][i]['text'], "label":"negative"})
  if len(tweets) >0:
    sid = tweets['search_metadata']['max_id_str']

  time.sleep(11)
