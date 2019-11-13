# CS410-final-project

# Target Dec 7, 2019

### Introduction

- Connect Twitter to our backend server (if possible)
- Script that can parse command line arguments or a file and provide sentiment analysis results
- Command line argument to provide an username to fetch recent tweets and provide sentiment analysis results(koric)
- Train our own sentiment analysis model (basic) (pranav) - DONE
- Use existing modules to import models for sentiment analysis (koric)
- PowerPoint and presentation
- Stock trend prediction (koric)


### Push Nov 12, 2019 - Pranav

- TrainModel.ipynb -> contains code that trained the model with 1.6m tweets. The pre-processing takes forever so I pickled the dataframe. Save time by unpacking that to get the data (df-cleaned-final.pickle has the cleaned up data)
- Test.ipynb -> contains demo code that can unpack and test the model
- shelve.model.db -> contains the trained model and TF-IDF vectors


### To do

- Use this trained model alongside a much more efficient library to analyze tweets by fetching them from Twitter via the API
- Build an application that provides an easy to use API for programs to interact with and get sentiment analysis for tweets (any text data should do)
- Use Pull requests to merge to master so we dont have overlapping and/or loose code by merging directly to master