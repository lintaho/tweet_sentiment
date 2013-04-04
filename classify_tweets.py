# Classifies a given set of tweets with a trained Naive Bayes classifier

from pymongo import MongoClient
import re
import time
import pickle
from svmutil import *

# MongoDB setup
start_time = time.time()
connection = MongoClient('localhost', 27017)
db = connection.data

svm = True
# Load classifier and features
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

# Load SVM classifier and features
def load_SVM(filec, filef):
    print 'loading SVM...'
    f = open(filec)
    classifier = svm_load_model(filec)
    f.close()
    print 'loading features...'
    f = open(filef)
    word_features = pickle.load(f)
    f.close()
    return classifier, word_features

# Select collections to classify
def select_collections(name, num_col, enum):
    t_cols = []
    if enum:
        for i in range(num_col):
            t_cols.append(name + str(i))
    else:
        t_cols.append(name)
    return t_cols

# Extract features from text
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

# Extract SVM features from text
def get_svm_features(tweets, featureList):
    sortedFeatures = sorted(featureList)
    map = {}
    feature_vector = []
    for t in tweets:
        map = {}
        for w in sortedFeatures:
            map[w] = 0
        tweet_words = t
        for word in tweet_words:
            if word in map:
                map[word] = 1
        values = map.values()
        feature_vector.append(values)
    return {'feature_vector': feature_vector}

# SVM special case
if not svm:
    classifier, word_features = load_classifier_and_features('classifier_NB_GOOD.pickle', 'features_NB_GOOD.pickle')
else:
    classifier, word_features = load_SVM('classifier_SVM.model', 'features_SVM.pickle')

times = ['121', '122', '123', '124', '125']

# For each day collection
for tstr in times:
    tweet_collections = select_collections('tech' + tstr, 0, False)
    print tweet_collections
    results_collection = db['results' + tstr]

    # classify each tweet collection
    print 'classifying...'
    for col in tweet_collections:
        pos_score, neg_score, neut_score = 0, 0, 0
        posl_score, negl_score, neutl_score = 0, 0, 0
        t = db[col].find()
        for tweet_object_index in range(t.count()):
            if tweet_object_index < t.count():
                text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", '', t[tweet_object_index]['text']).split())
                if not svm:
                    label = classifier.classify(extract_features(text.split(' ')))
                else:
                    test_feature_vector = get_svm_features([text.split(' ')], word_features)
                    p_labels, p_accs, p_vals = svm_predict([0] * len(test_feature_vector['feature_vector']), test_feature_vector['feature_vector'], classifier)
                    if p_labels[0] == 0.0:
                        label = 'negative'
                    else:
                        label = 'positive'
                    print label
                print str(tweet_object_index) + ' / ' + str(t.count())
                if label == 'positive':
                    pos_score += 1
                elif label == 'negative':
                    neg_score += 1
                else:
                    neut_score += 1
            else:
                break
            # results_collection.insert({'collection': col, 'time': t[tweet_object_index]['time'], 'pos': str(pos_score), 'neut': str(neut_score), 'neg': str(neg_score)})
            results_collection.insert({'collection': col, 'time': t[tweet_object_index]['time'], 'text': t[tweet_object_index]['text'], 'sent': label})

        print "Pos: " + str(pos_score) + " | Neg: " + str(neg_score) + " | Neut: " + str(neut_score)
        # results_collection.insert({'pos': str(pos_score), 'neg': str(neg_score)})object
        # print "Pos pattern: " + str(posl_score) + " | Neg pattern: " + str(negl_score) + " | Neut: " + str(neutl_score)

print 'classification time: ' + str(time.time() - start_time) + ' seconds'
