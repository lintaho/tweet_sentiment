from pymongo import MongoClient

# MongoDB setup
connection = MongoClient('localhost', 27017)
db = connection.data
collection = db['unlabeled']

col = db['labeled_data']
t = collection.find()
for tweet_object_index in range(t.count()):
    print t[tweet_object_index]['text']
    z = raw_input('label: ')
    if z == '1':
        label = 'positive'
    else:
        label = 'negative'
    col.insert({'text': t[tweet_object_index]['text'], 'label': label})
    print '-------------------------------'
    collection.remove(t[tweet_object_index]['_id'])
