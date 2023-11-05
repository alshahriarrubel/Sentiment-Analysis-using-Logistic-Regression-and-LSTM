# Name: AL SHAHRIAR RUBEL
# Email: ar2633@njit.edu

import torch

class Vocab:
    def __init__(self, embedding_dim=300):
        self.embedding_dim = embedding_dim
        self.embeddings = {}

        # Load the word embedding file glove.6B.300d.txt 
        self.load_embeddings('data/glove.6B.300d.txt')

    def load_embeddings(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                # Split the text of a line from glove.6B.300d.txt
                parts = line.strip().split(' ')
                # First part of the splitted text is a word
                word = parts[0]
                # Remaining parts of the splitted text is the corresponding vector
                embedding_vector = parts[1:]
                embedding_vector = torch.tensor(list(map(float, embedding_vector)))
                self.embeddings[word] = embedding_vector

    def get_embedding(self, word):
        if word in self.embeddings:
            return self.embeddings[word]
        else:
            return torch.zeros(self.embedding_dim)
