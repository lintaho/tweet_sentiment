Twitter Sentiment Analysis Comparison Tools
===============

These tools allow collection and training of three classifiers to find the polarity sentiment of tweets - 'positive' or 'negative.' To use, you need:

*MongoDB server* - can be on localhost

*NLTK* - http://nltk.org/

Usage
-----
*collect_tweets.py* - collect tweets and store into MongoDB

*train_classifier.py* - train classifier with args: 
- classifier (1=NB, 2=SVM, 3=ME)
- number of tweets to train on (default 20)
- OPTIONAL -k n, n-fold cross validation, n is the number of folds

example: 
<python train_classifier.py 2 5000 -k 4>
    
will train a Maximum Entropy classifier on 5000 tweets (1/2 pos, 1/2 neg) with 4-fold cross validation

Note- k-fold cross validation will not save the classifier. 


