# Classifies a given set of tweets with a trained Naive Bayes classifier

from pymongo import MongoClient
import re
import time
from pattern.en import polarity as pol
import pickle

start_time = time.time()
connection = MongoClient('localhost', 27017)
db = connection.data

def load_classifier_and_features(filec, filef):
    print 'loading classifier...'
    f = open(filec)
    classifier = pickle.load(f)
    f.close()
    print 'loading features...'
    f = open(filef)
    word_features = pickle.load(f)
    f.close()
    return classifier, word_features

def select_collections(name, num_col):
    t_cols = []
    for i in range(num_col):
        t_cols.append(name + str(i))
    return t_cols

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


classifier, word_features = load_classifier_and_features('classifier.pickle', 'features.pickle')
tweet_collections = select_collections('oscars', 2)
results_collection = db['results']

# classify each tweet collection per (hour?)
print 'classifying...'
for col in tweet_collections:
    pos_score, neg_score, neut_score = 0, 0, 0
    posl_score, negl_score, neutl_score = 0, 0, 0
    t = db[col].find()

    for tweet_object_index in range(t.count()):
        if tweet_object_index < 1000:
            text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", '', t[tweet_object_index]['text']).split())
            label = classifier.classify(extract_features(text.split(' ')))
            print str(tweet_object_index) + ' / ' + str(1000)
            if label == 'positive':
                pos_score += 1
            elif label == 'negative':
                neg_score += 1
            else:
                neut_score += 1
######################################################
            # if pol(text) > 0:
            #     posl_score += 1
            # elif pol(text) < 0:
            #     negl_score += 1
            # else:
            #     neutl_score += 1
######################################################
        else:
            break
    results_collection.insert({'collection': col, 'pos': str(pos_score), 'neut': str(neut_score), 'neg': str(neg_score)})

    print "Pos: " + str(pos_score) + " | Neg: " + str(neg_score) + " | Neut: " + str(neut_score)
    # print "Pos pattern: " + str(posl_score) + " | Neg pattern: " + str(negl_score) + " | Neut: " + str(neutl_score)

# add positives 1.0, negatives -1.0, score them or average them on some function
# plot these on a time series for a day

print 'classification time: ' + str(time.time() - start_time) + ' seconds'
