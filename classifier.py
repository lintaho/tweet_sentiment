import nltk
from pymongo import MongoClient
import re

connection = MongoClient('localhost', 27017)
db = connection.local
sad_col = db['neg_emoticons']
hap_col = db['pos_emoticons']
results_col = db['~results_superbowl_test1']
h= []
s = []
s=sad_col.find()
h = hap_col.find()
pos_tweets = []
neg_tweets = []
for tweet_object_index in range(s.count()):
  if tweet_object_index < 50:
    text =' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)","", s[tweet_object_index]['text']).split())
    neg_tweets.append((text, 'negative'))
  else:
    break
for tweet_object_index in range(h.count()):
  if tweet_object_index < 50:
#    text= re.search(r'/\w+/', h[tweet_object_index]['text']).group()
    text =' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)","", h[tweet_object_index]['text']).split())
    pos_tweets.append((text, 'positive'))
  else:
    break

#pos_tweets = [('I love this car', 'positive'),
 #             ('This view is amazing', 'positive')]
#neg_tweets = [('I do not like this car', 'negative'),
 #             ('This view is horrible', 'negative')]

tweets = []

for (words, sentiment) in pos_tweets + neg_tweets:
  words_filtered =[e.lower() for e in words.split() if len(e) >= 3]
  tweets.append((words_filtered, sentiment))
  bg = nltk.util.bigrams(words_filtered)
  tweets.append((bg, sentiment))
  tg = nltk.util.trigrams(words_filtered)
  tweets.append((tg, sentiment))
#test_tweets = tweets[:len(tweets)/5]
#tweets = tweets[len(tweets)/5:]

#print len(test_tweets)
print len(tweets)
def get_words_in_tweets(tweets):
  all_words = []
  for (words, sentiment) in tweets:
    all_words.extend(words)
  return all_words

def get_word_features(wordlist):
  wordlist = nltk.FreqDist(wordlist)
  word_features = wordlist.keys()
  return word_features

word_features = get_word_features(get_words_in_tweets(tweets))

#add to word_features from lexicon 
def extract_features(document):
  document_words = set(document)
  features = {}
  for word in word_features:
    if type(word) is unicode:
      txt = word
    else:
      txt = ' '.join(word)
    features['contains ' + txt] = (word in document_words)
  return features

training_set = nltk.classify.apply_features(extract_features, tweets)
#test_set = nltk.classify.apply_features(extract_features, test_tweets)
classifier = nltk.NaiveBayesClassifier.train(training_set)
print classifier.show_most_informative_features(100)

#print nltk.classify.accuracy(classifier, test_set)

t_cols = ["tweet_series_christmas0__01/31/13|||19:28:02",
"tweet_series_christmas1__01/31/13|||20:57:56",
"tweet_series_christmas2__01/31/13|||22:19:58",
"tweet_series_christmas3__01/31/13|||23:33:44",
"tweet_series_christmas4__02/01/13|||00:45:04",
"tweet_series_christmas5__02/01/13|||01:57:46",
"tweet_series_christmas6__02/01/13|||03:22:20",
"tweet_series_christmas7__02/01/13|||04:36:59",
"tweet_series_christmas8__02/01/13|||05:58:17",
"tweet_series_christmas9__02/01/13|||07:24:45",
"tweet_series_christmas10__02/01/13|||09:02:49",
"tweet_series_christmas11__02/01/13|||10:32:41",
"tweet_series_christmas12__02/01/13|||12:02:33",
"tweet_series_christmas13__02/01/13|||13:29:00",
"tweet_series_christmas14__02/01/13|||14:47:12",
"tweet_series_christmas15__02/01/13|||16:30:59",
"tweet_series_christmas16__02/01/13|||17:58:06",
"tweet_series_christmas17__02/01/13|||19:29:40",
"tweet_series_christmas18__02/01/13|||20:56:17",
"tweet_series_christmas19__02/01/13|||22:16:04",
"tweet_series_christmas20__02/01/13|||23:44:19",
"tweet_series_christmas21__02/02/13|||01:01:02",
"tweet_series_christmas22__02/02/13|||02:18:57",
"tweet_series_christmas23__02/02/13|||03:33:30"]

t_cols = ["tweet_superbowl0", "tweet_superbowl1", "tweet_superbowl2", "tweet_superbowl3", "tweet_superbowl4"]

#classify each tweet collection per (hour?)
#t_classify_col = db["tweet_series_christmas4__02/01/13|||00:45:04"]
for col in t_cols:
  t_classify_col = db[col]
  t=[]
  pos_score = 0
  neg_score = 0
  t=t_classify_col.find()
  for tweet_object_index in range(t.count()):
    if tweet_object_index < 70:
      text =' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)","", t[tweet_object_index]['text']).split())
      label = classifier.classify(extract_features(text.split(' '))) 
  #    print label, text
      if label is 'positive':
        pos_score +=1
      else:
        neg_score +=1
    else:
      break
  results_col.insert({"collection": col, "pos": str(pos_score), "neg": str(neg_score)})
  #print "Pos: " + str(pos_score) + " | Neg: " + str(neg_score)
#add positives 1.0, negatives -1.0, score them or average them on some function
#plot these on a time series for a day


