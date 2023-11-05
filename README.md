# Sentiment Analysis using Logistic Regression and LSTM

## Files:
- model.py
  <br> This file contains two classes for two models:
  - LogisticRegression
  - LSTM
 
- imdb_dataset.py
  - Read the dataset file ‘IMDB_Dataset.csv’ 
  - Select 15000 samples
  - Preprocessing such as tokenizing, removing stop words, stemming, Lemmatizing and word embedding
  - Append class label with embeddings

- vocab.py
  -	Load the word embedding file ‘glove.6B.300d.txt’
  -	Split the text of each line 
  - Extract first part of the splitted text that is a word
  - Extract remaining parts of the splitted text that is the corresponding vector of 300 dimensions

- train_logreg.py
  -	This file is for training the logistic regression model
  -	It loads the IMDB dataset
  -	Split the dataset into train and test sets
  -	Create the data loaders
  -	Create the model
  -	Define the loss function and optimizer
  -	Train the model
  -	Evaluate the model on the test set

- train_lstm.py
  -	This file is for training the LSTM model
  -	It loads the IMDB dataset
  -	Split the dataset into train and test sets
  -	Create the data loaders
  -	Create the model
  -	Define the loss function and optimizer
  -	Train the model
  -	Evaluate the model on the test set

- IMDB_Dataset.csv
  -	It contains 25000 samples for sentiment analysis
  -	Two classes: Negative, Positive

- Requirements.txt
  -	It contains the commands for environment settings and necessary libraries installation
- glove.6B.300d.txt
  -	This file is for word embedding

## How to run:
* Command to Train and Test Logistic Regression
-     Python train_logreg.py   

* Command to Train and Test LSTM
-     Python train_lstm.py   

