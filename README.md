Twitter Sentiment Analysis Comparison Tools
===============

These tools allow collection and training of three classifiers to find the polarity sentiment of tweets - 'positive' or 'negative.' To use, you need:

*MongoDB server* - can be on localhost

*NLTK* - http://nltk.org/

Usage
-----

In *collect_tweets.py* and *tweetstreamer.py* fill in the following snippet with your app key, app secret, oauth token, and oauth token secret from [Twitter](https://dev.twitter.com/).

t = Twython(app_key='APPKEY',
            app_secret='APPSECRET',
            oauth_token='OAUTHTOKEN',
            oauth_token_secret='OAUTHTOKENSECRET'
            )

*collect_tweets.py* - collect tweets and store into MongoDB

*train_classifier.py* - train classifier with args: 
- classifier (1=NB, 2=SVM, 3=ME)
- number of tweets to train on (default 20)
- OPTIONAL -k n, n-fold cross validation, n is the number of folds

Example: 
>python train_classifier.py 2 5000 -k 4

will train a Maximum Entropy classifier on 5000 tweets (1/2 pos, 1/2 neg) with 4-fold cross validation

Note- k-fold cross validation will not save the classifier. 

*tweetstreamer.py* - streams tweets throughout a trading day.

*find_best_words.py* - iterates through a list of column names and finds the highest correlated keywords that lead to stock price change within a trading day.

*correlate_and_plot.py* - prints correlation between a list of tweet sentiments and prices within trading day, and plots them.