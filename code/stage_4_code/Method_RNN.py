from code.base_class.method import method
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
from torch import nn
import torch as th
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm as tq
import matplotlib.pyplot as plt

class Method_RNN(method, nn.Module):
    def __init__(self, mName, mDescription, vocab_size, embedding_dim, hidden_dim, output_dim):
        nn.Module.__init__(self)
        method.__init__(self, mName, mDescription)

        self.epoch = 20
        self.learning_rate = 0.001

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # Original
        # self.embedding = nn.Embedding(self.vocab_size, embedding_dim=self.embedding_dim)
        # self.rnn = nn.RNN(self.embedding_dim, self.hidden_dim, batch_first=True)
        # self.fc = nn.Linear(self.hidden_dim, self.output_dim)

        # LSTM
        # Embedding layer
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        # LSTM layer
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=2, batch_first=True, dropout=0.5,
                            bidirectional=True)

        # Dropout
        self.dropout = nn.Dropout(0.5)

        # Fully connected layer
        self.fc1 = nn.Linear(self.hidden_dim * 2, 128)  # Adjust for bidirectional output
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, self.output_dim)

    # def forward(self, x):
    #     embedded = self.embedding(x)  # text: [batch size, seq length]
    #     output, hidden = self.lstm(embedded)  # output: [batch size, seq length, hidden dim], hidden: [1, batch size, hidden dim]
    #     hidden = hidden.squeeze(0)  # hidden: [batch size, hidden dim]
    #     output = self.fc(hidden)  # output: [batch size, output dim]
    #     return output

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        output = self.dropout(output[:, -1, :])  # Use the output of the last sequence step
        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)
        return output

    # Initialize
    def train_data(self, train_loader, validation_loader, test_loader, device):
        print(len(train_loader))
        print(len(validation_loader))
        print(len(test_loader))
        #
        self.to(device)

        # Train Batch Loss
        train_losses = []
        validation_losses = []

        # Define Adam Optimizer
        optimizer = th.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Define Criterion
        criterion = nn.BCEWithLogitsLoss()

        # Training Process
        for epoch in range(self.epoch):
            self.train()
            train_batch_losses = []

            for encoding, label in tq(train_loader):
                encoding, label = encoding.to(device), label.to(device)

                # Zero Gradient Descent
                optimizer.zero_grad()

                # Forward Pass
                outputs = self(encoding)

                # Calculate Loss Function
                loss = criterion(outputs, label)

                # Backward Pass
                loss.backward()

                # Update model's parameters
                optimizer.step()

                # Record Batch Loss
                batch_loss = loss.item()
                train_batch_losses.append(batch_loss)

            epoch_loss = np.mean(train_batch_losses)
            train_losses.append(epoch_loss)

            self.eval()
            # Testing Accuracy for every epoch
            correct, total = 0, 0

            with th.no_grad():  # No gradient computation within this block
                validation_batch_losses = []
                for images, labels in validation_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = self(images)  # Get model predictions (logits)

                    loss = criterion(outputs, labels)  # Calculate loss
                    validation_batch_loss = loss.item()  # Convert loss to a Python number
                    validation_batch_losses.append(validation_batch_loss)  # Record batch loss

                    # Convert logits to probabilities using sigmoid
                    predicted_probs = th.sigmoid(outputs).squeeze()  # Apply sigmoid and remove any extra dimension
                    predicted_labels = (predicted_probs > 0.5).float()  # Apply threshold to determine predicted labels

                    total += labels.size(0)
                    correct += (predicted_labels == labels).sum().item()  # Calculate the number of correct predictions

                validation_epoch_loss = np.mean(validation_batch_losses)  # Calculate mean validation loss for the epoch
                validation_losses.append(validation_epoch_loss)

                validation_accuracy = correct / total  # Calculate validation accuracy percentage
                # print(f"Epoch {epoch}: Validation Accuracy: {validation_accuracy:.2f}.")

        self.eval()
        test_accuracy, test_precision, test_f1, test_recall = self.test_data(test_loader, device)
        print(f"The final test accuracy is {test_accuracy}")

        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(validation_losses, label='Validation Loss')
        plt.title('Training and Validation Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def test_data(self, test_loader, device):
        correct, total = 0, 0
        all_predicts = []
        all_labels = []

        evaluate = Evaluate_Accuracy()

        with th.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)

                _, predicted = th.max(outputs, 1)

                all_predicts.extend(predicted.view(-1).cpu().numpy())
                all_labels.extend(labels.view(-1).cpu().numpy())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        evaluate_accuracy = correct / total
        evaluate_precision = evaluate.evaluate_precision(all_labels, all_predicts)
        evaluate_f1 = evaluate.evaluate_f1(all_labels, all_predicts)
        evaluate_recall = evaluate.evalulate_recall(all_labels, all_predicts)

        return evaluate_accuracy, evaluate_precision, evaluate_f1, evaluate_recall
