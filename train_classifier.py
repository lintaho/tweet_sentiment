# Trains and saves a classifier based with unigram, bigram, and trigram features extract from tweets
# Also uses as features words from a lexicon containing neutral words

import nltk
from pymongo import MongoClient
import re
import time
import pickle

start_time = time.time()
connection = MongoClient('localhost', 27017)
db = connection.local
sad_col = db['neg_emoticons']
hap_col = db['pos_emoticons']

h, s = [], []
s = sad_col.find()
h = hap_col.find()

pos_tweets, neg_tweets = [], []

for tweet_object_index in range(s.count()):
    if tweet_object_index < 6281:
        text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", '', s[tweet_object_index]['text']).split())
        neg_tweets.append((text, 'negative'))
    else:
        break

for tweet_object_index in range(h.count()):
    if tweet_object_index < 6281:
        text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", '', h[tweet_object_index]['text']).split())
        pos_tweets.append((text, 'positive'))
    else:
        break

tweets = []

def add_XX_features():
    f = open('words.tff')
    p, n, neut = [], [], []
    for line in f.readlines():
        t = line.split(' ')
        word = (t[2])[6:]
        sent = (t[5])[14:].strip()
       # stemmed = (t[4])[9:]  # TODO
        if sent == 'positive':
            p.append(word)
        if sent == 'negative':
            n.append(word)
        if sent == 'neutral':
            neut.append(word)

    tweets.append((p, 'positive'))
    tweets.append((n, 'negative'))
    tweets.append((neut, 'netural'))

def add_bigrams():
    for (words, sentiment) in pos_tweets + neg_tweets:
        words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
        tweets.append((words_filtered, sentiment))
        bg = nltk.util.bigrams(words_filtered)
        tweets.append((bg, sentiment))
        tg = nltk.util.trigrams(words_filtered)
        tweets.append((tg, sentiment))

def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
        all_words.extend(words)
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

add_XX_features()
add_bigrams()
word_features = get_word_features(get_words_in_tweets(tweets))

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        if type(word) is unicode or type(word) is str:
            txt = word
        else:
            txt = ' '.join(word)
        features['contains ' + txt] = word in document_words
    return features

#save the classifier so don't have to retrain later
def save_classifier(document, cfier):
    f = open(document, 'wb')
    pickle.dump(cfier, f)
    f.close()
    print 'classifier saved in ' + document

def save_features(document, feats):
    f = open(document, 'wb')
    pickle.dump(feats, f)
    f.close()
    print 'features saved in ' + document

# test_tweets = tweets[:len(tweets)/5]
# tweets = tweets[len(tweets)/5:]
training_set = nltk.classify.apply_features(extract_features, tweets)
# test_set = nltk.classify.apply_features(extract_features, test_tweets)

print 'training ' + str(len(tweets))
classifier = nltk.NaiveBayesClassifier.train(training_set)

print classifier.show_most_informative_features(100)
print 'training time: ' + str(time.time() - start_time) + ' seconds'
#print nltk.classify.accuracy(classifier, test_set)

save_classifier('classifier.pickle', classifier)
save_features('features.pickle', word_features)
