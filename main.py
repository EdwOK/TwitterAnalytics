import os
import pickle
import re
import sys
from operator import itemgetter

import tweepy
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

# twitter oauth
consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = 's'
owner = ''
owner_id = ''


def load_user_tweets(user_name):
    with open('%s.pickle' % user_name, 'rb') as in_file:
        load_tweets = pickle.load(in_file)

    return load_tweets


def save_user_tweets(user_name, user_tweets):
    with open('%s.pickle' % user_name, 'wb') as out_file:
        pickle.dump(user_tweets, out_file)


def get_api():
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)
    return api


def is_user_exist(api, user_name):
    try:
        api.get_user(user_name)
        return True
    except:
        return False


def get_user_tweets(api, user_name):
    user_tweets = []

    new_tweets = api.user_timeline(screen_name=user_name, count=200, tweet_mode='extended')
    user_tweets.extend(new_tweets)

    tweet_tail = user_tweets[-1].id - 1

    while len(new_tweets) > 0:
        new_tweets = api.user_timeline(screen_name=user_name, count=200, max_id=tweet_tail, tweet_mode='extended')
        user_tweets.extend(new_tweets)

        tweet_tail = user_tweets[-1].id - 1

    return user_tweets


def process_tweets(user_tweets):
    processed_tweets = []

    for tweet in user_tweets:
        process_tweet = tweet.full_text

        try:
            process_tweet = re.sub(r"https?\S+", '', process_tweet)
        except:
            process_tweet = process_tweet

        processed_tweets.append(process_tweet)

    return processed_tweets


def main(user_name):
    if os.path.exists('%s.pickle' % user_name):
        user_tweets = load_user_tweets(user_name=user_name)
    else:
        api = get_api()

        if not is_user_exist(api=api, user_name=user_name):
            sys.exit("error: user does not exist!")
            return

        print('Start scraping ...')
        user_tweets = get_user_tweets(api=api, user_name=user_name)
        save_user_tweets(user_name=user_name, user_tweets=user_tweets)
        print('Finish scraping and saving ...')

    print('Start training ...')

    processed_tweets = process_tweets(user_tweets=user_tweets)
    newsgroups = fetch_20newsgroups(subset='all', data_home='.')

    text_clf = Pipeline([
        ('vect', CountVectorizer(stop_words='english')),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(max_iter=10))
    ])

    text_clf = text_clf.fit(newsgroups.data, newsgroups.target)
    prediction = text_clf.predict(processed_tweets)

    print('Finish training ...')

    top_categories = {}

    for doc, category in zip(processed_tweets, prediction):
        cat_key = newsgroups.target_names[category]

        if cat_key in top_categories:
            top_categories[cat_key] += 1
        else:
            top_categories[cat_key] = 1

    top_categories = sorted(top_categories.items(), key=itemgetter(1), reverse=True)

    for category, count in top_categories[:10]:
        print(category, count)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        user_name = input('Please typing twitter username: ')
        main(user_name=user_name)
    elif len(sys.argv) == 2:
        main(user_name=sys.argv[1])
    else:
        sys.exit("error: incorrect number of input arguments!")
        sys.exit(0)
