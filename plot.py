# Plots a set of positive/negative tweets onto a time series based on their ratio

from pymongo import MongoClient
import sys
import matplotlib.pyplot as plt
from numpy import *

connection = MongoClient('localhost', 27017)
db = connection.local

results_col = db[str(sys.argv[1])]

scores = []

t = results_col.find()
for tweet_index in range(t.count()):
    pos = t[tweet_index]['pos']
    neg = t[tweet_index]['neg']
    ratio = float(pos) / float(neg)
    scores.append(ratio)

a = array(scores)
a /= a.max()

plt.plot(a)
plt.axis([0, t.count(), 0, 1])
plt.show()
