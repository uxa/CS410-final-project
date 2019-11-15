import re 
import tweepy 
from tweepy import OAuthHandler 
from textblob import TextBlob


class twitter_api(object):
    def __init__(self): 
        ''' 
        Class constructor or initialization method. 
        '''
        # keys and tokens from the Twitter Dev Console 
        consumer_key = 'gLRNOAhuVMPDvtr5aOvYqZ6Ze'
        consumer_secret = '6n9F6Ieedd97SrtvZFiRvf5k5uognXDEYTUabsnIidKHH3PaDA'
        access_token = '919706112261312512-fP82zHMs27OeeIsVtVXpNrVEt2CBSBH'
        access_token_secret = 'zPUl78nX5hNONyBy6ei943TTKgonzN0JXhLfteYVQ5YKS'
  
        # attempt authentication 
        try: 
            # create OAuthHandler object 
            self.auth = OAuthHandler(consumer_key, consumer_secret) 
            # set access token and secret 
            self.auth.set_access_token(access_token, access_token_secret) 
            # create tweepy API object to fetch tweets 
            self.api = tweepy.API(self.auth) 
        except: 
            print("Error: Authentication Failed")

    def get_tweets(self, query, count = 1):
        tweets = []

        try:
            fetched_tweets = self.api.search(q = query, count = count)  
            for tweet in fetched_tweets:
                parsed_tweet = tweet.text
                if tweet.retweet_count > 0: 
                    if parsed_tweet not in tweets: 
                        tweets.append(parsed_tweet) 
                else: 
                    tweets.append(parsed_tweet)
            return tweets
        except tweepy.TweepError as e:  
            print("Error : " + str(e))
