# Sentiment Analysis using Logistic Regression and LSTM

Files:
•	model.py
o	This file contains two classes for two models: 
	LogisticRegression 
	LSTM

•	imdb_dataset.py
o	Read the dataset file ‘IMDB_Dataset.csv’ 
o	Select 15000 samples
o	Preprocessing such as tokenizing, removing stop words, stemming, Lemmatizing and word embedding
o	Append class label with embeddings

•	vocab.py
o	Load the word embedding file ‘glove.6B.300d.txt’
o	Split the text of each line 
o	Extract first part of the splitted text that is a word
o	Extract remaining parts of the splitted text that is the corresponding vector of 300 dimensions

•	train_logreg.py
o	This file is for training the logistic regression model
o	It loads the IMDB dataset
o	Split the dataset into train and test sets
o	Create the data loaders
o	Create the model
o	Define the loss function and optimizer
o	Train the model
o	Evaluate the model on the test set

•	train_lstm.py
o	This file is for training the LSTM model
o	It loads the IMDB dataset
o	Split the dataset into train and test sets
o	Create the data loaders
o	Create the model
o	Define the loss function and optimizer
o	Train the model
o	Evaluate the model on the test set

•	IMDB_Dataset.csv
o	It contains 25000 samples for sentiment analysis
o	Two classes: Negative, Positive
![image](https://github.com/alshahriarrubel/Sentiment-Analysis-using-Logistic-Regression-and-LSTM/assets/24860187/d2127fd7-3133-432e-a79c-cf693ccecead)
