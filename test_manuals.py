# MongoDB setup
from pymongo import MongoClient
import pickle
from svmutil import *
import re

connection = MongoClient('localhost', 27017)
db = connection.data
col = db['labeled_data']

svm = False
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

pos_score, neg_score, neut_score = 0, 0, 0
t = col.find()
acc = 0
for tweet_object_index in range(t.count()):
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
        if label == t[tweet_object_index]['label']:
            acc += 1
    print str(tweet_object_index) + ' / ' + str(t.count())
    if label == t[tweet_object_index]['label']:
        acc += 1
print str(acc) + '/' + str(t.count())
