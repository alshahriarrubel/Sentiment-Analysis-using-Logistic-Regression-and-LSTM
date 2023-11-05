# Name: AL SHAHRIAR RUBEL
# Email: ar2633@njit.edu

import torch
from torch.utils.data import DataLoader, random_split
import sklearn
from imdb_dataset import IMDBDataset
from vocab import Vocab
from model import LogisticRegression, LSTM
from sklearn.metrics import accuracy_score, classification_report
import nltk
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')

def train_LSTM(model, train_loader, optimizer, criterion, epochs):
    for epoch in range(epochs):
        for batch in train_loader:
            x, y = batch
            y = y.float()

            # Preprocess the input data
            x = x.permute(1, 0, 2)  # (batch_size, seq_len, embedding_dim)

            # Forward pass
            outputs = model(x)
            outputs = torch.mean(outputs, dim=0).view(-1)

            # Calculate the loss
            loss = criterion(outputs, y)

            # Backward pass and update the model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoch:", epoch, "Loss:", loss.item())

if __name__ == "__main__":
    # Load the IMDB dataset
    print('Loading dataset...\n')
    vocab = Vocab()

    # Split the dataset into train and test sets
    print('Splitting dataset into train and test sets...\n')
    dataset = IMDBDataset("IMDB_Dataset.csv", vocab, stop_words, stemming=True)
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])

    # Create the data loaders
    print('creating dataloader...\n')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    ##### LSTM
    print("\nLSTM model.....\n")
    # Create the model
    print('Creating model...\n')
    model = LSTM(vocab.embedding_dim,10, 1)

    # Define the loss function and optimizer
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    print('Training the model...\n')
    train_LSTM(model, train_loader, optimizer, criterion, epochs=100)

    # Evaluate the model on the test set
    print("Evaluating...")
    correct = 0
    total = 0
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            y = y.float()

            x = x.permute(1, 0, 2)

            outputs = model(x)
            outputs = torch.mean(outputs, dim=0).view(-1)
            predictions = (outputs > 0.5).float()

            correct += (predictions == y).sum().item()
            total += y.size(0)
            true_labels.extend(y.cpu().numpy())
            predicted_labels.extend(predictions.cpu().numpy())

    print(f"Accuracy: {(correct / total):.2f}")
    accuracy = accuracy_score(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels, target_names=['Negative', 'Positive'])
    print(f'\tLSTM - Accuracy: {accuracy:.2f}')
    print(f'\n\tLSTM - Classification Report:\n')
    print(report)

    report2 = classification_report(true_labels, predicted_labels, target_names=['Negative', 'Positive'], output_dict=True)
    overall_metrics = report2['weighted avg']
    print(f'\tLSTM - Overall Precision: {overall_metrics["precision"]:.2f}')
    print(f'\tLSTM - Overall Recall: {overall_metrics["recall"]:.2f}')
    print(f'\tLSTM - Overall F1-Score: {overall_metrics["f1-score"]:.2f}\n')

