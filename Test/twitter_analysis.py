import sklearn
import pickle
import nltk
from nltk.corpus import stopwords # nltk.download('stopwords') before importing
from nltk.stem import PorterStemmer
from twitter_api import twitter_api
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
import re
import preprocessor as p
from nltk.tokenize import WordPunctTokenizer
from bs4 import BeautifulSoup
import csv
tok = WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, pat2))

stemmer = PorterStemmer()





def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    stripped = re.sub(combined_pat, '', souped)
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped
    letters_only = re.sub("[^a-zA-Z']+", " ", clean)
    lower_case = letters_only.lower()
    words = tok.tokenize(lower_case)
    output = (" ".join(words)).strip()
    output = output.replace(" ' ", "")
    return output

def predict(tweet, model):
    return model.predict(vector.transform([tweet_cleaner(tweet)]))

def check_model(score):
    if score > 0:
        result = "positvie"
    else:
        result = "negative"
    return result

def check_polarity(score):
    result = 2
    if score > 0:
        result = "positvie"
    elif score == 0:
        result = "neutrual"
    else:
        result = "negative"
    return result

with open('Pickled data/model.pickle', 'rb') as f:
    model_LR = pickle.load(f)
    
with open('Pickled data/naive-bayes.pickle', 'rb') as f2:
    model_NB = pickle.load(f2)
    
with open('Pickled data/vector.pickle', 'rb') as f3:
    vector = pickle.load(f3)


twitter_username = input("What is the username you would like to evaluate?:")
count_tweets = int(input("How many tweets from the latest would you like to evaluate?:"))

api = twitter_api()
tweets = api.get_tweets(twitter_username,count_tweets)


# print(tweets[0])
# print(tweet_cleaner(tweets[0]))
count = 1
for tweet in tweets:
    print("Tweet number", str(count)+":")
    print(tweet_cleaner(tweet))
    lr = predict(tweet, model = model_LR)
    lr_result = check_model(lr[0])
    print("Using Logistic Regression this tweet is", lr_result)
    nb = predict(tweet, model = model_NB)
    nb_result = check_model(nb[0])
    print("Using NB this tweet is", lr_result)
    tweet = tweet_cleaner(tweet)
    analysis = TextBlob(tweet)
    score_model = check_polarity(analysis.sentiment.polarity)
    print("Using Text Blob this tweet is", score_model)
    print("")
    count += 1

    

