# Name: AL SHAHRIAR RUBEL
# Email: ar2633@njit.edu

import csv
import numpy as np
import torch
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from torch.utils.data import Dataset
import torch.nn.functional as F

class IMDBDataset(Dataset):
    def __init__(self, csv_file, vocab, stop_words, stemming=True, lemmatizing=True):
        self.data = []
        # Open the dataset file
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            next(reader)
            max_tokens = 300
            idx = 0
            for row in reader:
                idx =idx+1
                # Take 15000 samples from dataset
                if idx > 15000:
                    break
                review = row[0]
                # Tokenization
                tokens = word_tokenize(review)
                # Remove stop words
                tokens = [token for token in tokens if token not in stop_words]
                # Stemming
                if stemming:
                    tokens = [nltk.PorterStemmer().stem(token) for token in tokens]
                # Lemmatizing
                elif lemmatizing:
                    tokens = [nltk.WordNetLemmatizer().lemmatize(token) for token in tokens]
                if len(tokens) > max_tokens:
                    tokens = tokens[:max_tokens]
                # Create embeddings
                embeddings = torch.zeros(max_tokens, vocab.embedding_dim)
                # Convert tokens to word embeddings
                for i, token in enumerate(tokens):
                    embeddings[i] = vocab.get_embedding(token)
                # Append embedddings and class label in data
                data_label = (row[1] == 'positive')
                self.data.append((embeddings, int(data_label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
