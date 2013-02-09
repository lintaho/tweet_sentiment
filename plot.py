from pymongo import MongoClient
import sys
import matplotlib.pyplot as plt

connection = MongoClient('localhost', 27017)
db = connection.local

results_col = db[str(sys.argv[1])]

scores = []

t = results_col.find()
for tweet_index in range(t.count()):
  pos = t[tweet_index]['pos']
  neg = t[tweet_index]['neg']

  ratio = float(pos)/float(neg)
  scores.append(ratio)

plt.plot(scores)
plt.axis([0,24, 0, 2])
plt.show()

