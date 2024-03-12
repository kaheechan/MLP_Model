from code.base_class.method import method
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
from torch import nn
import torch as th
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm as tq
import matplotlib.pyplot as plt

class Method_CNN(method, nn.Module):
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # Initialize Parameters
        self.learning_rate = 0.001
        self.epochs = 10

        self.mName = mName
        self.mDescription = mDescription

        # ORL
        if self.mName == 'ORL':
            self.conv_layer_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)
            self.batch_norm_1 = nn.BatchNorm2d(16)
            self.pool_layer_1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv_layer_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
            self.batch_norm_2 = nn.BatchNorm2d(32)
            self.pool_layer_2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv_layer_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
            self.batch_norm_3 = nn.BatchNorm2d(64)
            self.pool_layer_3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.dropout = nn.Dropout(0.5)
            # self.flatten = nn.Flatten()
            # self.flatten_layer = nn.Linear(in_features=7*7*10, out_features=10)
            self.fc_1 = nn.Linear(64 * 14 * 11, 1024)
            self.fc_2 = nn.Linear(1024, 40)

        # MNIST
        elif self.mName == 'MNIST':
            self.conv_layer_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
            self.batch_norm_1 = nn.BatchNorm2d(16)
            self.pool_layer_1 = nn.MaxPool2d(2, 2)
            self.conv_layer_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
            self.batch_norm_2 = nn.BatchNorm2d(32)
            self.dropout = nn.Dropout(0.5)
            self.fc_1 = nn.Linear(6272, 120)
            self.fc_2 = nn.Linear(120, 10)

        elif self.mName == 'CIFAR':
            self.conv_layer_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
            self.batch_norm_1 = nn.BatchNorm2d(32)
            self.pool_layer_1 = nn.MaxPool2d(2, 2)
            self.conv_layer_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
            self.batch_norm_2 = nn.BatchNorm2d(64)
            self.pool_layer_2 = nn.MaxPool2d(2, 2)
            self.conv_layer_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
            self.batch_norm_3 = nn.BatchNorm2d(128)
            self.pool_layer_3 = nn.MaxPool2d(2, 2)
            self.fc_1 = nn.Linear(128 * 4 * 4, 256)
            self.fc_2 = nn.Linear(256, 10)

    def forward(self, x):
        if self.mName == 'ORL':
            x = self.conv_layer_1(x)
            x = self.batch_norm_1(x)
            x = F.relu(x)
            x = self.pool_layer_1(x)

            x = self.conv_layer_2(x)
            x = self.batch_norm_2(x)
            x = F.relu(x)
            x = self.pool_layer_2(x)

            x = self.conv_layer_3(x)
            x = self.batch_norm_3(x)
            x = F.relu(x)
            x = self.pool_layer_3(x)

            x = x.view(x.size(0), -1)
            x = self.dropout(x)

            x = self.fc_1(x)
            x = F.relu(x)
            x = self.fc_2(x)

        elif self.mName == 'MNIST':
            x = self.conv_layer_1(x)
            x = self.batch_norm_1(x)
            x = F.relu(x)
            x = self.pool_layer_1(x)

            x = self.conv_layer_2(x)
            x = self.batch_norm_2(x)
            x = F.relu(x)
            x = x.view(x.size(0), -1)
            x = self.dropout(x)

            x = self.fc_1(x)
            x = F.relu(x)
            x = self.fc_2(x)

        elif self.mName == 'CIFAR':
            x = self.conv_layer_1(x)
            x = self.batch_norm_1(x)
            x = F.relu(x)
            x = self.pool_layer_1(x)

            x = self.conv_layer_2(x)
            x = self.batch_norm_2(x)
            x = F.relu(x)
            x = self.pool_layer_2(x)

            x = self.conv_layer_3(x)
            x = self.batch_norm_3(x)
            x = F.relu(x)
            x = self.pool_layer_3(x)

            x = x.view(-1, 128 * 4 * 4)

            x = self.fc_1(x)
            x = F.relu(x)
            x = self.fc_2(x)

        return x

    def train_data(self, train_loader, validation_loader, test_loader, device):
        # Define Adam Optimizer
        optimizer = th.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Define Criterion
        criterion = nn.CrossEntropyLoss()

        # Loss Parameter
        train_losses = []
        validation_losses = []

        for epoch in range(self.epochs):
            print(f"Running Epoch: {epoch}...")
            train_batch_losses = []
            for images, labels in tq(train_loader):
                images, labels = images.to(device), labels.to(device)

                # Zero Gradient Descent
                optimizer.zero_grad()

                # Forward Pass
                outputs = self(images)

                # Calculate the loss function
                loss = criterion(outputs, labels)

                # Backward Pass
                loss.backward()

                # Update model's parameters
                optimizer.step()

                # Record Batch Loss
                batch_loss = loss.item()
                train_batch_losses.append(batch_loss)

            #
            epoch_loss = np.mean(train_batch_losses)
            train_losses.append(epoch_loss)

            #
            self.eval()
            # Testing Accuracy for every epoch
            correct, total = 0, 0

            with th.no_grad():
                validation_batch_losses = []
                for images, labels in validation_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = self(images)
                    loss = criterion(outputs, labels)

                    validation_batch_loss = loss.item()
                    validation_batch_losses.append(validation_batch_loss)

                    _, predicted = th.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                validation_epoch_loss = np.mean(validation_batch_loss)
                validation_losses.append(validation_epoch_loss)

            validation_accuracy = correct / total
            print(f"Epoch {epoch}: The accuracy is: {validation_accuracy}.")

        self.eval()
        test_accuracy, test_precision, test_f1, test_recall = self.test_data(test_loader, device)
        print(f"The final test accuracy is {test_accuracy},\n"
              f"test precision is {test_precision},\n"
              f"test f1 is {test_f1},\n"
              f"test recall is {test_recall}.\n")

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