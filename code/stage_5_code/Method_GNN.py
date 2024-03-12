from code.base_class.method import method
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch as th
import matplotlib.pyplot as plt
from code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy

class Method_GNN(method, nn.Module):
    def __init__(self, mName, mDescription, nFeatures, nClasses):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # Class Informations
        self.mName = mName
        self.mDescription = mDescription
        self.nNodes = nFeatures
        self.nClasses = nClasses

        self.learning_rate = 0.001
        self.epochs = 150

        # Cora
        if self.mName == 'cora':
            self.initial_conv = GCNConv(nFeatures, 32)
            self.middle_conv_1 = GCNConv(32, 16)
            self.final_conv = GCNConv(16, nClasses)
        # Citeseer
        elif self.mName == 'citeseer':
            self.initial_conv = GCNConv(nFeatures, 32)
            self.middle_conv_1 = GCNConv(32, 16)
            self.final_conv = GCNConv(16, nClasses)
        # Pubmed
        elif self.mName == 'pubmed':
            self.initial_conv = GCNConv(nFeatures, 32)
            self.middle_conv_1 = GCNConv(32, 64)
            self.middle_conv_2 = GCNConv(64, 32)
            self.final_conv = GCNConv(32, nClasses)

    def forward(self, x, edge_index):
        if self.mName == 'cora':
            x = F.relu(self.initial_conv(x, edge_index))
            x = F.dropout(x, training=self.training)
            x = F.relu(self.middle_conv_1(x, edge_index))
            x = F.dropout(x, training=self.training)
            x = F.relu(self.final_conv(x, edge_index))
            x = F.log_softmax(x, dim=1)

        elif self.mName == 'citeseer':
            x = F.relu(self.initial_conv(x, edge_index))
            x = F.dropout(x, training=self.training)
            x = F.relu(self.middle_conv_1(x, edge_index))
            x = F.dropout(x, training=self.training)
            x = F.relu(self.final_conv(x, edge_index))
            x = F.log_softmax(x, dim=1)

        elif self.mName == 'pubmed':
            x = F.relu(self.initial_conv(x, edge_index))
            x = F.dropout(x, training=self.training)
            x = F.relu(self.middle_conv_1(x, edge_index))
            x = F.dropout(x, training=self.training)
            x = F.relu(self.middle_conv_2(x, edge_index))
            x = F.dropout(x, training=self.training)
            x = F.relu(self.final_conv(x, edge_index))
            x = F.log_softmax(x, dim=1)

        return x

    def train_data(self, data):
        optimizer = th.optim.Adam(self.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        train_losses = []
        validation_losses = []
        validation_accuracies = []

        for epoch in range(self.epochs):

            # Train the data
            self.train()

            # Zero Gradient Descent
            optimizer.zero_grad()

            # Forward Pass
            outputs = self(data.x, data.edge_index)

            # Loss Function
            loss = criterion(outputs[data.train_mask], data.y[data.train_mask])

            # Backward Pass
            loss.backward()

            # Update Model Parameters
            optimizer.step()

            # Record Batch Loss
            batch_loss = loss.item()
            train_losses.append(batch_loss)

            #
            self.eval()

            with th.no_grad():
                out = self(data.x, data.edge_index)
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask]).item()
                pred = out.argmax(dim=1)
                correct = (pred[data.val_mask] == data.y[data.val_mask]).float().sum()
                val_acc = correct / data.val_mask.sum()

            validation_losses.append(val_loss)
            validation_accuracies.append(val_acc.item())

            print(
                f'Epoch: {epoch + 1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(validation_losses, label='Validation Loss')
        plt.plot(validation_accuracies, label='Validation Accuracy')
        plt.title('Training, Validation Losses and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Loss/Accuracy')
        plt.legend()
        plt.show()

    def test_data(self, data):
        self.eval()  # Ensure the model is in evaluation mode
        with th.no_grad():  # No need to compute gradients
            out = self(data.x, data.edge_index)  # Forward pass with the whole graph
            pred = out.argmax(dim=1)  # Get the index of the max log-probability
            y_true = data.y[data.test_mask].cpu().numpy()  # True labels
            y_pred = pred[data.test_mask].cpu().numpy()  # Predicted labels

            # Calculate accuracy
            correct = (pred[data.test_mask] == data.y[data.test_mask]).float().sum()
            acc = correct / data.test_mask.sum()

            # Calculate precision, recall, and F1 score
            evaluate_accuracy = Evaluate_Accuracy()

            precision = evaluate_accuracy.evaluate_precision(y_true, y_pred)
            recall = evaluate_accuracy.evalulate_recall(y_true, y_pred)
            f1 = evaluate_accuracy.evaluate_f1(y_true, y_pred)

            # Print the metrics
            print(f'Test Accuracy: {acc:.4f}')
            print(f'Precision: {precision:.4f}')
            print(f'Recall: {recall:.4f}')
            print(f'F1 Score: {f1:.4f}')












