# Trains and saves a classifier based with unigram, bigram, and trigram features extract from tweets
# Also uses as features words from a lexicon containing neutral words

import nltk
from pymongo import MongoClient
import random
import re
import time
import collections
import pickle
from nltk.corpus import stopwords
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from svmutil import *

def remove_stopwords(text):
    return [w for w in text if not w in stopwords.words('english')]

start_time = time.time()
connection = MongoClient('localhost', 27017)
db = connection.local
sad_col = db['neg_emoticons']
hap_col = db['pos_emoticons']

h, s = [], []
s = sad_col.find()
h = hap_col.find()

pos_tweets, neg_tweets = [], []

if len(sys.argv) > 2:
    count = int(sys.argv[2])
else:
    count = 20

for tweet_object_index in range(s.count()):
    if tweet_object_index < count:
        text = ' '.join(remove_stopwords(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\S+:\/\/\w+)", '', s[tweet_object_index]['text']).split()))
        neg_tweets.append((text, 'negative'))
    else:
        break

for tweet_object_index in range(h.count()):
    if tweet_object_index < count:
        text = ' '.join(remove_stopwords(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", '', h[tweet_object_index]['text']).split()))
        pos_tweets.append((text, 'positive'))
    else:
        break


# negWords, posWords = [], []
# for t in neg_tweets:
#     negWords.append(t[0].split())
# for t in pos_tweets:
#     posWords.append(t[0].split())

# # print negWords
# negWords = list(itertools.chain(*negWords))
# posWords = list(itertools.chain(*posWords))



def get_freqs(posWords, negWords):
    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in posWords:
        # print word
        word_fd.inc(word.lower())
        cond_word_fd['pos'].inc(word.lower())
    for word in negWords:
        word_fd.inc(word.lower())
        cond_word_fd['neg'].inc(word.lower())

    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    # print pos_word_count, neg_word_count, total_word_count

    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score
    return word_scores

def find_best_words(word_scores, number):
    best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words

def best_word_features(words):
    return dict([('contains ' + word, True) for word in words if word in best_words])

# best_words = find_best_words(word_scores, 1000)
# bwf = best_word_features(best_words)

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

def add_ngrams(tweetlist):
    t_list = []
    for (words, sentiment) in tweetlist:
        words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
        t_list.append((words_filtered, sentiment))
        bg = nltk.util.bigrams(words_filtered)
        t_list.append((bg, sentiment))
        tg = nltk.util.trigrams(words_filtered)
        t_list.append((tg, sentiment))
    return t_list

def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
        all_words.extend(words)
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features


tweets = pos_tweets + neg_tweets
num_trained = len(tweets)
random.shuffle(tweets)
test_tweets = tweets[:len(tweets) / 5]
tweets = tweets[len(tweets) / 5:]
tweets = add_ngrams(tweets)
test_tweets = add_ngrams(test_tweets)
del neg_tweets
del pos_tweets

#add_XX_features()
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

def get_svm_features(tweets, featureList):
    sortedFeatures = sorted(featureList)
    map = {}
    # print tweets
    feature_vector = []
    labels = []
    for t in tweets:
        label = 0
        map = {}
        for w in sortedFeatures:
            map[w] = 0
        tweet_words = t[0]
        tweet_opinion = t[1]
        for word in tweet_words:
            if word in map:
                map[word] = 1
        values = map.values()
        feature_vector.append(values)
        if tweet_opinion == 'positive':
            label = 0
        elif tweet_opinion == 'negative':
            label = 1
        labels.append(label)
    return {'feature_vector': feature_vector, 'labels': labels}

def k_fold_validation(k, tweets):
    sets = []
    acc = 0.0
    for i in range(k):
        sets.append(tweets[(len(tweets) / k * i):(len(tweets) / k * i + len(tweets) / k)])
        # print len(tweets[(len(tweets) / k * i):(len(tweets) / k * i + len(tweets) / k)])
    for i in range(k):
        training_tweets = []
        test_tweets = sets[i]
        for j in range(k):
            if i != j:
                training_tweets.extend(sets[j])
        # print len(training_tweets), len(test_tweets)
        training_set = nltk.classify.apply_features(extract_features, training_tweets)
        test_set = nltk.classify.apply_features(extract_features, test_tweets)
        del training_tweets
        del test_tweets
        print 'training ' + str(len(training_set))
        classifier = nltk.NaiveBayesClassifier.train(training_set)
        #  print test_set[0][1]
        pos_score, neg_score, neut_score = 0, 0, 0
        print 'finding accuracy...'
        for i, tweet in enumerate(test_set):
            label = classifier.classify(tweet[0])
            if label == tweet[1]:
                pos_score += 1
            else:
                neg_score += 1
        # del training_set
        acc = float(pos_score) / float(len(test_set))
        print acc
        save_classifier('classifier_xfold' + str(acc) + '.pickle', classifier)
        #print nltk.classify.accuracy(classifier, test_set)



training_set = nltk.classify.apply_features(extract_features, tweets)
test_set = nltk.classify.apply_features(extract_features, test_tweets)


def train_svm():
    print 'training svm classifier with ' + str(num_trained) + ' tweets'
    result = get_svm_features(tweets, word_features)
    problem = svm_problem(result['labels'], result['feature_vector'])
    param = svm_parameter('-q')

    # param.cross_validation = 1
    # param.nr_fold = 10

    param.kernel_type = LINEAR
    classifier = svm_train(problem, param)
    # svm_save_model('classifier_SVM_x.model', classifier)
    test_feature_vector = get_svm_features(test_tweets, word_features)
    # print test_feature_vector
    p_labels, p_accs, p_vals = svm_predict(test_feature_vector['labels'], test_feature_vector['feature_vector'], classifier)
    pf, pr, pp = calc_prec_recall_svm(test_feature_vector['labels'], p_labels, 1)
    nf, nr, np = calc_prec_recall_svm(test_feature_vector['labels'], p_labels, 0)
    print 'pos precision: ', str(pp)
    print 'pos recall: ', str(pr)
    print 'pos fscore: ', str(pf)
    print 'neg precision: ', str(np)
    print 'neg recall: ', str(nr)
    print 'neg fscore: ', str(pf)
    return classifier

def train_naive_bayes():
    print 'training naive bayes classifier with ' + str(num_trained) + ' tweets'
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    return classifier

def train_maxent():
    print 'training max ent classifier with ' + str(num_trained) + ' tweets'
    classifier = nltk.classify.maxent.MaxentClassifier.train(training_set, 'GIS', trace=3, encoding=None, labels=None, sparse=True, gaussian_prior_sigma=0, max_iter=10)
    return classifier

def calc_prec_recall_svm(ty, pv, sign):
        tp = 0.0
        fp = 0.0
        fn = 0.0
        #sign 1 = pos, 0 = neg
        for i in range(len(ty)):
            if pv[i] == sign and ty[i] == sign:
                    tp += 1
            else:
                if pv[i] == sign and ty[i] != sign:
                    fp += 1
                else:
                    if pv[i] != sign and ty[i] == sign:
                        fn += 1
        if tp + fp == 0:
            precision = 0
        else:
            precision = float(tp) / float(tp + fp)
        if tp + fn == 0:
            recall = 0
        else:
            recall = float(tp) / float(tp + fn)
        if precision + recall == 0:
            fscore = 0
        else:
            fscore = 2 * precision * recall / (precision + recall)
        return (fscore, recall, precision)

def calc_prec_recall(classifier):
    referenceSets = collections.defaultdict(set)
    testSets = collections.defaultdict(set)

    for i, (features, label) in enumerate(test_set):
        # print features, i
        referenceSets[label].add(i)
        predicted = classifier.classify(features)
        testSets[predicted].add(i)

    pp = nltk.metrics.precision(referenceSets['positive'], testSets['positive'])
    pr = nltk.metrics.recall(referenceSets['positive'], testSets['positive'])
    np = nltk.metrics.precision(referenceSets['negative'], testSets['negative'])
    nr = nltk.metrics.recall(referenceSets['negative'], testSets['negative'])
    if pp and pr:
        pos_fscore = 2 * (pp * pr) / (pr + pp)
    else:
        pos_fscore = 0
    if np and nr:
        neg_fscore = 2 * (np * nr) / (nr + np)
    else:
        neg_fscore = 0
    print 'pos precision: ', pp
    print 'pos recall: ', pr
    print 'pos fscore: ', str(pos_fscore)
    print 'neg precision: ', np
    print 'neg recall: ', nr
    print 'neg fscore: ', str(neg_fscore)

def show_stats(classifier):
    classifier.show_most_informative_features(10)
    print 'training time: ' + str(time.time() - start_time) + ' seconds'
    # print 'accuracy: ' + str(nltk.classify.accuracy(classifier, test_set))

if str(sys.argv[1]) == '1':
    classifier = train_svm()

    svm_save_model('classifier_SVM.model', classifier)
    save_features('features_SVM.pickle', word_features)
    #m = svm_load_model('libsvm.model')

elif str(sys.argv[1]) == '2':
    classifier = train_maxent()
    calc_prec_recall(classifier)
    show_stats(classifier)
    save_classifier('classifier_ME.pickle', classifier)
    save_features('features.pickle', word_features)
else:
    classifier = train_naive_bayes()
    calc_prec_recall(classifier)
    show_stats(classifier)
    save_classifier('classifier_NB.pickle', classifier)
    save_features('features_NB.pickle', word_features)

# k_fold_validation(4, tweets)
