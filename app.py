
'''

CS410 Final Project, UIUC - Fall 2019
Team Wolfram
@Authors: 
    - Pranav Velamakanni (pranavv2@illinois.edu)
    - Tarik Koric (koric1@illinois.edu)

Summary:
    Requirements: Python 3+
    Modules: see README.md for a complete list

This project aims to provide sentiment analysis on live tweets fetched from Twitter. 
The prediction is based on 3 models trained by us using a set of 1.6 million tweets.
All the below models have been trained and pickled to a file which is imported here.
Refer to TrainModel.ipynb for the training code.

# LogisticRegression - Accuracy ~ 77%
# Naive-Bayes - Accuracy ~ 76%
# Neural Network (single layer with 100 units) - Accuracy ~ 71%

'''

import logging
import argparse
import pickle
import os
import time
from collections import Counter

from tweepy import OAuthHandler
from tweepy import StreamListener
import tweepy
from nltk.corpus import stopwords # nltk.download('stopwords') before importing
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt

### Pickled models
base_pickle_dir = 'Pickled data' # This directory contains the pickled models
models = ['LR.pickle', 'nn.pickle', 'naive-bayes.pickle'] # File names of the models
models_path = [*map(lambda file : os.path.join(base_pickle_dir, file), models)]

class Model:

    '''
    This class creates objects for the pre-trained models.
    '''

    ## TF-IDF vector required to transform the tweet
    # Vector specifications: Max features - 10,000, Ngram range - (1, 2)
    vector = None

    def __init__(self, model = None):

        self.models = dict(zip(['LogisticRegression', 'NeuralNetwork', 'NaiveBayes'], models_path))

        if not model or model not in self.models:
            model = 'LogisticRegression'
        
        self.model = self.import_model(self.models.get(model))

        if not Model.vector:
            Model.vector = self.init_vector()

    def import_model(self, model_path):
        '''
        Loads the corresponding model from the pickle.
        '''

        with open(model_path, 'rb') as md:
            return pickle.load(md)

    def init_vector(self):
        '''
        Load the trained TF-IDF vector from pickle.
        '''

        with open(os.path.join(base_pickle_dir, 'vector.pickle'), 'rb') as vc:
            return pickle.load(vc)

    def label_prediction(self, prediction):
        '''
        Converts integer predictions to string.
        '''

        return 'Positive' if prediction[-1] == 1 else 'Negative'

    def predict(self, data):
        '''
        Clean, transform a tweet and predict sentiment.
        '''

        if isinstance(data, str):
            return self.label_prediction(self.model.predict(Model.vector.transform([Twitter.clean(data)])))
        
        result = list()
        for tweet in data:
            result.append(self.label_prediction(self.model.predict(Model.vector.transform([Twitter.clean(data)]))))

        return predict

    
class Twitter:

    '''
    This class creates an object that enables interaction with the Twitter API.
    '''

    def __init__(self):
        ''' 
        Initializes a Twitter object with authentication.
        '''

        # Keys and tokens from the Twitter Dev Console 
        consumer_key = 'gLRNOAhuVMPDvtr5aOvYqZ6Ze'
        consumer_secret = '6n9F6Ieedd97SrtvZFiRvf5k5uognXDEYTUabsnIidKHH3PaDA'
        access_token = '919706112261312512-fP82zHMs27OeeIsVtVXpNrVEt2CBSBH'
        access_token_secret = 'zPUl78nX5hNONyBy6ei943TTKgonzN0JXhLfteYVQ5YKS'
  
        # Attempt authentication 
        try: 
            # Create OAuthHandler object 
            self.auth = OAuthHandler(consumer_key, consumer_secret) 
            # Set access token and secret 
            self.auth.set_access_token(access_token, access_token_secret) 
            # Create tweepy API object to fetch tweets 
            self.api = tweepy.API(self.auth) 
        except: 
            logging.fatal("Error: Authentication Failed")

    @classmethod
    def clean(cls, tweet):
        '''
        Classmethod: clean tweet by removing stopwords, hashtags, @ mentions, websites and perform stemming.
        '''

        stemmer = PorterStemmer()
        stage1 = [word for word in tweet.lower().split() if word not in stopwords.words('english')] # stopword removal
        stage2 = [word[1:] if word.startswith('#') else word for word in stage1] # Hashtag symbol removal
        stage3 = [stemmer.stem(word) for word in stage2 if not any([word.startswith('@'), word.startswith('http'), word.startswith('www')])] # @ mentions and websites removal and stemming
        return ' '.join(stage3)

    def get_tweets_by_user(self, user, count = 10):
        '''
        Fetch tweets for a certain user (username).
        Max count supported by Twitter is 200.
        '''

        try:
            tweet_object = self.api.user_timeline(screen_name = user, count = count)
            return [*map(lambda x : x.text, tweet_object)]
        except tweepy.TweepError as e:
            logging.exception('Failed to fetch tweets: {}'.format(str))

    def search_tweets(self, query, count = 10):
        '''
        Search for tweets matching a query.
        #NOTE: This does not perform an exhaustive search.
        '''

        try:
            tweet_object = self.api.search(query = query, count = count)
            return [*map(lambda x : x.text, tweet_object)]
        except tweepy.TweepError as e:
            logging.exception('Failed to fetch tweets: {}'.format(str(e)))


class Classifier:

    '''
    Creates an object that loads all the models to perform analysis.
    '''

    def __init__(self):

        self.LR = Model('LogisticRegression')
        self.NB = Model('NaiveBayes')
        self.NN = Model('NeuralNetwork')
        self.models = {'LogisticRegression': self.LR,
                        'NaiveBayes': self.NB,
                        'NeuralNetwork': self.NN}

    def weighted_average(self, data):
        '''
        The prediction of the Classifier is weighted to give models with higher
        accuracy more weight.
        Ranking based on accuracy, 1 - LR, 2 - NB, 3 - NN
        '''
        # Weights correspond to the ranking above.
        weights = {'LogisticRegression': 0.40,
                    'NaiveBayes': 0.35,
                    'NeuralNetwork': 0.25} 
        
        total = {'Positive' : 0, 'Negative': 0}

        for model, score in data.items():
            total[score] += weights[model]

        return (max(total, key = total.get), total[max(total, key = total.get)])

    def predict(self, text, generate_summary = True):
        '''
        Predicts the sentiment of a tweet using all the imported models.
        '''

        predictions = dict()
        if isinstance(text, str):
            for name, model in self.models.items():
                predictions[name] = model.predict(text)
            return self.get_summary(predictions) if generate_summary else predictions

    def get_summary(self, predictions):
        '''
        Using the raw predictions, generates a weighted pretty-printed summary.
        '''
        result = str()

        for name, score in predictions.items():
            result += '{}: {}\n'.format(name, score)

        final_score = self.weighted_average(predictions)
        result += 'Prediction: {} with a probability of {}%\n'.format(final_score[0], final_score[-1]*100)
        return result

    def visualize(self, results, title = None, save_to_file = False):
        '''
        Provides a pie chart to summarize the predicted sentiments
        '''

        final_scores_combined = Counter([self.weighted_average(result)[0] for result in results])

        labels = final_scores_combined.keys()
        sizes = final_scores_combined.values()
        colors = ['lightcoral' if l == 'Negative' else 'yellowgreen' for l in labels]
        explode = (0.1, 0)

        plt.pie(sizes, labels = labels, explode = explode, colors = colors, shadow = True, autopct = '%1.1f%%')
        plt.axis('equal')

        if title:
            plt.title(title)

        if save_to_file:
            plt.savefig('TweetSummaryPlot.png')
            logging.info('Plot saved as TweetSummaryPlot.png')
        else:
            plt.show()

    def process_data(self, data, user = None, save_to_file = False, visualize = False):
        '''
        Saves tweets to file and optionally provides a visual summary of predictions.
        '''

        predictions, results = list(), list()

        for tweet in data:
            print('Tweet: {}'.format(tweet))
            prediction = self.predict(tweet, generate_summary = False)
            result = self.get_summary(prediction)
            predictions.append(prediction)
            results.append(result)
            print(result)
        
        if save_to_file:
            with open('TweetAnalysis-{}.txt'.format(user), 'w') as file:
                for tweet, result in zip(data, results):
                    file.write(tweet + '\n')
                    file.write(result)
            logging.info('File saved as TweetAnalysis-{}.txt'.format(user))

        if visualize:
            self.visualize(predictions, title = user, save_to_file = save_to_file)


class TwitterStreamListener(tweepy.StreamListener):
    '''
    Creates a Twitter stream object.
    '''

    def __init__(self, topic = None, classifier = None, save_to_file = False, time_limit = 20, visualize = False):

        self.predictions = list()
        self.limit = time_limit
        self.start = time.time()

        if not classifier:
            self.model = Classifier()
        else:
            self.model = classifier

        self.save_to_file = save_to_file
        self.visualize = visualize
        self.topic = topic

        if self.save_to_file:
            self.open_file = open('TweetStreamAnalysis.txt', 'w')

        super(TwitterStreamListener, self).__init__()

    def on_status(self, data):
        '''
        Process the incoming stream of tweets. 
        '''

        if (time.time() - self.start) < self.limit:

            prediction = self.model.predict(str(data.text), generate_summary = False)
            summary = self.model.get_summary(prediction)
            self.predictions.append(prediction)

            print('Tweet: {}'.format(data.text))
            print(summary)

            if self.save_to_file:
                self.open_file.write('Tweet: {}\n'.format(data.text))
                self.open_file.write(summary)

            time.sleep(0.25) # This sleep ensures that the stdout is not flooded with tweets.
            return True
        else:
            if self.save_to_file:
                self.open_file.close()
                logging.info('File saved as TweetStreamAnalysis.txt')

            if self.visualize:
                self.model.visualize(self.predictions, title = self.topic, save_to_file = self.save_to_file)

            return False
    
    def on_error(self, status):
        '''
        Handle the error status.
        '''

        logging.error('Terminating program, error: {}'.format(status))
        return False

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Tweet sentiment analyzer')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--user', '-u', type=str, default=None, help='Twitter username to fetch tweets')
    group.add_argument('--stream', nargs='+', type=str, default=None, help='Stream a list of topics from Twitter')
    parser.add_argument('--file', action='store_true', default=False, help='Store tweets and analysis from stream to file')
    parser.add_argument('--visualize', action='store_true', default=False, help='Provides a pie chart with a summary of predictions')
    parser.add_argument('--count', '-c', type=lambda x : 200 if int(x) > 200 else abs(int(x)), default=10, help='Number of tweets to fetch')
    parser.add_argument('--time', type=int, default=20, help='Time to stream a topic')

    args = parser.parse_args()

    if args.user:
        # Initialize twitter object
        tw_connection = Twitter()
        tweets = tw_connection.get_tweets_by_user(user = args.user, count = args.count)

        # Initialize classifier
        model = Classifier()
        model.process_data(data = tweets, user = args.user, save_to_file = args.file, visualize = args.visualize)

    if args.stream:
        tw_connection = Twitter()

        # Initialize stream object
        streamListener = TwitterStreamListener(topic = args.stream, save_to_file = args.file, time_limit = args.time, visualize = args.visualize)
        stream = tweepy.Stream(auth = tw_connection.api.auth, listener = streamListener)
        stream.filter(track = args.stream)

