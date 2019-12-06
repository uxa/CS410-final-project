#### CS410 Final Project, Fall 2019
#### Team Wolfram
#### Pramav Velamakanni (pranavv2@illinois.edu), Tarik Koric (koric1@illinois.edu)

### Introduction

The aim of this project is to design a system that can provide live sentiment analysis on a stream of tweets from Twitter. This is achieved by training 3 models on a dataset with 1.6 million tweets.

The CSV file containing the tweets can be downloaded from - https://www.kaggle.com/kazanova/sentiment140/download

To be able to retain the original state of the trained model and to provide predictions quickly, the trained models and vectors are saved to a binary file using Pickle. These models and vectors are loaded into memory when the application is executed. Live tweets are streamed from Twitter and passed to the models to predict the sentiment. The application supports several arguments which are discussed below.

The final prediction is decided based on the individual scores by the 3 models. Models with higher accuracy are given a higher weight for the final prediction probability.

### Tools used in this project

- `Python 3.7`
- `Jupyter` - notebooks used to train and test the models
- `Pickle` - used to save the trained models and vectors as binary files
- `Pandas, NumPy` - load and manipulate data using DataFrames
- `NLTK` - used in data pre-processing and cleaning
- `Scikit-learn` - machine learning algorithm toolkit
- `Tweepy` - Twitter API to stream live tweets
- `Matplotlib` - tool to visualize the results

### Set up the enviornment to use this application

#### Method 1 (pip)

Please ensure you have Python 3 installed

The following command can be run to install all the dependencies (using `pip`) needed for this app to run.

`pip install --upgrade jupyter pandas numpy nltk scikit-learn tweepy matplotlib`

#### Method 2 (Conda environment)

If you have [Anaconda](https://www.anaconda.com/distribution/) installed, the dependencies can be installed to a custom enviornment (ideal if you have other projects using different versions of the libraries)

Create the environment:  `conda create -n TeamWolfram python=3.7`

Activate the environment:  `conda activate TeamWolfram`

From this project workspace execute: `pip install --requirement requirements.txt`

#### Download stopwords package

From the terminal, run:  `python -c "import nltk; nltk.download('stopwords')"`

### Files in this workspace

- `app.py` - Main application file that interacts with the tweets and the models
- `TrainModel.ipynb` - This notebook contains the pre-processing and model training
- `requirements.txt` - File containing the Python requirements for this project
- `Test/` directory (Misc: tests performed while testing and tweaking the application)
    - `Test.ipynb` - Notebook containing test code to unpack and load the model for predictions
    - `twitter_analysis.py` - Initial tests using the Twitter API and the trained models
    - `twitter_api.py` - Initial tests setting up the Twitter API
    - `TweetStreamAnalysis.txt` - Test file containing tweets saved after running a stream
    - `TweetSummaryPlot.png` - Test pie chart generated from the stream predictions
- `Pickled data/` directory
    - `LR.pickle` - Pickled trained Logistic regression model
    - `naive-bayes.pickle` - Pickled trained Naive Bayes model
    - `nn.pickle` - Pickled trained Neural Network model
    - `vector.pickle` - Pickled TF-IDF vector to transform the data

### Models and achieved accuracy

- `Logistic Regression` - 77%
- `Naive Bayes` - 76%
- `Neural Network` - 71%

### Data and pre-processing

- 1.6 m individual tweets with a 1 (Positive) or 0 (Negative) label
- Data cleaning involved the following steps
    - Convert the tweet to lowercase, remove stopwords
    - Remove the hashtag symbol (`#`)
    - Remove `@` mentions, websites
    - Perform stemming
- TF-IDF vector with the following specs
    - 10000 max features
    - 1-2 Ngrams
    - L2 normalization 

### Modules used

- `Data processing`: Numpy, Pandas, NLTK
- `Analysis`: Scikit-learn
- `Model packaging`: Pickle
- `Twitter API`: Tweepy

### How to use the app

- `app.py` is a command line app that supports the following arguments
    - Tweets from a specific user
        - `--user` or `-u` - username of the user to fetch tweets from (example - elonmusk (without the `@`))
        - `--count` or `-c` - number of tweets to fetch and analyze (example - 5, defaults to 10)
    - Stream tweets for a list of topics
        - `--stream` - list of topics to fetch live tweets from Twitter and perform analysis (example - "trump" "Tesla" "Penguins")
        - `--time` - total duration of the stream in seconds (example - 10, defaults to 20)
        - `--file` - save the tweets and performed analysis to a file named `TweetStreamAnalysis.txt` in the current workspace
        - `--visualize` - visualizes the predictions using a pie chart. Saves to file when `--file` flag is used

*Please Note: This app provides default API access keys to use this application for testing purposes. It is however, recommended to change these to values in `app.py` for extensive usage. Instructions to generate new keys can be found [here](https://developer.twitter.com/en/docs/basics/authentication/guides/access-tokens) if you would like to change these keys.*

### Examples

#### Using Streams with multiple topics for a period of 10 seconds
```
❯ python app.py --stream "trump" "Pittsburgh Penguins" "NHL" --time 10 --file                                                                                                                                                               ─╯
Tweet: RT @EgSophie: For reference: this is Jovi Val, on the left posing in front of a swastika and giving the Nazi salute, and on the right in hi…
LogisticRegression: Positive
NaiveBayes: Positive
NeuralNetwork: Positive
Prediction: Positive with a probability of 100.0%

Tweet: A year long physical 🤔  maybe plastic surgery 😅😂🤣 President Trump began phase one of his annual physical at Walter… https://t.co/HKpuHODmLV
LogisticRegression: Negative
NaiveBayes: Positive
NeuralNetwork: Positive
Prediction: Positive with a probability of 60.0%

Tweet: RT @michellemalkin: The EB-5 racket is a ghastly selling out of US citizenship to the highest foreign bidders. John Miano &amp; I exposed the s…
LogisticRegression: Positive
NaiveBayes: Positive
NeuralNetwork: Negative
Prediction: Positive with a probability of 75.0%

Tweet: RT @ThePlumLineGS: Important exchange here: https://t.co/WxhLeO2AWx
LogisticRegression: Positive
NaiveBayes: Positive
NeuralNetwork: Positive
Prediction: Positive with a probability of 100.0%

Tweet: RT @maggie_pdx: @brianklaas Every day I wonder what other 'favors' Trump has attempted collection on in service of his 2020 reelection.
LogisticRegression: Positive
NaiveBayes: Negative
NeuralNetwork: Positive
Prediction: Positive with a probability of 65.0%

Tweet: RT @PressSec: Very well said! If the dems had the votes they wouldn’t be prolonging this charade. They’re just working with their partners…
LogisticRegression: Positive
NaiveBayes: Positive
NeuralNetwork: Positive
Prediction: Positive with a probability of 100.0%
```

#### Picking a specific user and fetching last 20 tweets
```
❯ python app.py --user elonmusk --count 20                                                                                                                                                                                                  ─╯
Tweet: @farrxy @Ford I’d be way too embarrassed to put that on a Tesla. It’s like a kid’s drawing.
LogisticRegression: Negative
NaiveBayes: Positive
NeuralNetwork: Negative
Prediction: Negative with a probability of 65.0%

Tweet: @Ford Congratulations on the Mach E! Sustainable/electric cars are the future!! Excited to see this announcement fr… https://t.co/vlFHJeb7Mt
LogisticRegression: Positive
NaiveBayes: Negative
NeuralNetwork: Positive
Prediction: Positive with a probability of 65.0%

Tweet: @flcnhvy Exactly! Well said.
LogisticRegression: Positive
NaiveBayes: Negative
NeuralNetwork: Negative
Prediction: Negative with a probability of 60.0%

Tweet: @cleantechnica Surprisingly common
LogisticRegression: Positive
NaiveBayes: Negative
NeuralNetwork: Positive
Prediction: Positive with a probability of 65.0%
```

#### Streaming topics and visualizing the results
```
❯ python app.py --stream "tesla" "elon musk" "cybertruck" --time 15 --visualize                                                                                                                                                                             ─╯
Tweet: AHAHAHAHAHAHAHAHA but real talk, the #cybertruck is my favorite new car to be released since probably the ND miata.
LogisticRegression: Positive
NaiveBayes: Positive
NeuralNetwork: Negative
Prediction: Positive with a probability of 75.0%

Tweet: Metro Boomin turn this hoe into a mosh pit, Tesla build got my flying like a cockpit
LogisticRegression: Negative
NaiveBayes: Positive
NeuralNetwork: Positive
Prediction: Positive with a probability of 60.0%

Tweet: RT @jfagone: His company SpaceX could have minimized the interference by simply painting the satellites black. But they didn’t do that. htt…
LogisticRegression: Positive
NaiveBayes: Negative
NeuralNetwork: Negative
Prediction: Negative with a probability of 60.0%

Tweet: RT @CNN: A Ford executive, reacting to a video of Tesla's all-electric Cybertruck winning a tug-of-war against a Ford F-150, challenged Tes…
LogisticRegression: Positive
NaiveBayes: Positive
NeuralNetwork: Positive
Prediction: Positive with a probability of 100.0%
```

![Pie chart](https://i.imgur.com/PV7xaDK.png)