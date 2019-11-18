from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob 
import pandas as pd
import re
import preprocessor as p
from nltk.tokenize import WordPunctTokenizer
from bs4 import BeautifulSoup
import csv
tok = WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, pat2))



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

def rating(num):
    rate = int(num)
    output = ""
    if rate == 4:
        output = 'pos'
    elif rate == 0:
        output = 'neg'
    return output


df = pd.read_csv('train_online.csv', encoding = "ISO-8859-1", names=['score','id','date','query','name','tweet'])

train=df.sample(frac=0.8,random_state=200) 
test=df.drop(train.index)


twitter_text = train[['tweet', 'score']]
twitter_text = twitter_text.sample(n=10000)
twitter_text = twitter_text.reset_index(drop=True)
new_tweets = twitter_text['tweet'].apply(tweet_cleaner)
clean_scores = twitter_text['score'].map(rating)
new = pd.concat([new_tweets, clean_scores], axis=1)
new.to_csv('out_train.csv', index = False, header=None)


twitter_text = train[['tweet', 'score']]
twitter_text = twitter_text.sample(n=1000)
twitter_text = twitter_text.reset_index(drop=True)
new_tweets = twitter_text['tweet'].apply(tweet_cleaner)
clean_scores = twitter_text['score'].map(rating)
new = pd.concat([new_tweets, clean_scores], axis=1)
new.to_csv('out_test.csv', index = False, header=None)



correct = 0
total = 0

with open('out_train.csv', 'r') as fp:
    cl = NaiveBayesClassifier(fp, format="csv")




with open('out_test.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')

    for row in readCSV:
        score = cl.classify(row[0])
        if score == row[1]:
            correct += 1
        total += 1

print(correct/total)