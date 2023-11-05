# Sentiment Analysis using Logistic Regression and LSTM

Files:
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
