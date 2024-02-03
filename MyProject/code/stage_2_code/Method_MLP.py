from code.base_class.method import method
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy

import torch as th
import numpy as np
import matplotlib.pyplot as plt

from torch import nn

# @dataclass
# class Method_MLP(method, nn.Module):
#     mName: str = None
#     mDescription: str = None
#
#     def __post_init__(self):
#         method.__post_init__(self)
#         nn.Module.__init__(self)
#
#         # Define Default Parameters
#         self.learning_rate = 0.001
#         self.epochs = 500
#
#         # Define Function Here
#         self.linear_layer_1 = nn.Linear(4, 4)
#         self.activation_function_1 = nn.ReLU()
#         self.linear_layer_2 = nn.Linear(4, 2)
#         self.activation_function_2 = nn.Softmax(dim=1)
#
#     def forward_propagation(self, x):
#         hidden_layer = self.activation_function_1(self.linear_layer_1(x))
#         return self.activation_function_2(self.linear_layer_2(hidden_layer))
#
#     def train_data(self, X, y):
#         # Note: self.parameters() still works because it is inherited from nn.Module
#         optimizer = th.optim.Adam(self.parameters(), lr=self.learning_rate)
#         loss_function = nn.CrossEntropyLoss()
#
#         evaluate_accuracy = Evaluate_Accuracy('training evaluator', '')
#
#         for epoch in range(self.epochs):
#             x = th.FloatTensor(np.array(X))
#             y_pred = self.forward_propagation(x)
#             y_true = th.LongTensor(np.array(y))
#
#             loss = loss_function(y_pred, y_true)
#
#             optimizer.zero_grad()
#             loss.backward()
#
#             optimizer.step()
#
#             if epoch % 100 == 0:
#                 evaluate_accuracy.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
#                 print('Epoch: ', epoch, 'Accuracy: ', evaluate_accuracy.evaluate_data(), 'Loss: ', loss.item())
#
#     def test_data(self, X):
#         x = th.FloatTensor(np.array(X))
#         y_pred = self.forward(x)
#         return y_pred.max(1)[1]
#
#     def run_data(self, X_train, y_train, X_test, y_test):
#         print('Method running...')
#         print('--Start training...')
#         self.train_data(X_train, y_train)
#         print('--Start testing...')
#         y_pred = self.test_data(X_test)
#         return {'pred_y': y_pred, 'true_y': y_test}

class Method_MLP(method, nn.Module):
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # Define Default Parameters
        self.learning_rate = 0.001
        self.epochs = 1500

        # Define Function Here
        self.linear_layer_1 = nn.Linear(784, 100)
        self.activation_function_1 = nn.ReLU()
        self.linear_layer_2 = nn.Linear(100, 500)
        self.activation_function_2 = nn.ReLU()
        self.linear_layer_3 = nn.Linear(500, 100)
        self.activation_function_3 = nn.ReLU()
        self.final_linear_layer = nn.Linear(100, 10)
        self.final_activation_function = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.activation_function_1(self.linear_layer_1(x))
        x = self.activation_function_2(self.linear_layer_2(x))
        x = self.activation_function_3(self.linear_layer_3(x))
        return self.final_activation_function(self.final_linear_layer(x))

    def train_data(self, X_train, y_train, X_test, y_test):
        # Note: self.parameters() still works because it is inherited from nn.Module
        optimizer = th.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()

        train_evaluate_accuracy = Evaluate_Accuracy('training evaluator', '')
        test_evaluate_accuracy = Evaluate_Accuracy('testing evaluator', '')

        train_points = []
        train_epochs = []

        test_points = []
        test_epochs = []

        for epoch in range(self.epochs):
            self.train()
            x_train = th.FloatTensor(np.array(X_train))
            y_train_pred = self.forward(x_train)

            y_train_true = th.LongTensor(np.array(y_train))

            train_loss = loss_function(y_train_pred, y_train_true)


            optimizer.zero_grad()
            train_loss.backward()
            # test_loss.backward()

            optimizer.step()

            # Evaluating Validation Data
            self.eval()
            with th.no_grad():
                x_test = th.FloatTensor(np.array(X_test))
                y_test_pred = self.forward(x_test)
                y_test_true = th.LongTensor(np.array(y_test))
                test_loss = loss_function(y_test_pred, y_test_true)

            if epoch % 10 == 0:
                train_loss_point = train_loss.item()
                test_loss_point = test_loss.item()

                train_evaluate_accuracy.data = {'true_y': y_train_true, 'pred_y': y_train_pred.max(1)[1]}
                test_evaluate_accuracy.data = {'true_y': y_test_true, 'pred_y': y_test_pred.max(1)[1]}
                print('Epoch: ', epoch, 'Train Accuracy: ', train_evaluate_accuracy.evaluate_accuracy(), 'Train Loss: ', train_loss_point)
                train_points.append(train_loss_point)
                train_epochs.append(epoch)

                print('Epoch: ', epoch, 'Test Accuracy: ', test_evaluate_accuracy.evaluate_accuracy(), 'Test Loss: ', test_loss_point)
                test_points.append(test_loss_point)
                test_epochs.append(epoch)

        return train_epochs, train_points, test_epochs, test_points

    def test_data(self, X):
        x = th.FloatTensor(np.array(X))
        y_pred = self.forward(x)
        return y_pred.max(1)[1]

    def run_data(self, X_train, y_train, X_test, y_test):
        print('Method running...')
        print('--Start training...')
        train_epochs, train_points, test_epochs, test_points = self.train_data(X_train, y_train, X_test, y_test)
        print('--Start testing...')
        y_pred = self.test_data(X_test)
        y_test = th.Tensor(y_test.tolist())

        # Plot the Learning Curve
        plt.plot(train_epochs, train_points)
        plt.plot(test_epochs, test_points)
        plt.xlabel('Epochs')
        plt.ylabel('Losses')
        plt.title('Learning Curve')
        plt.show()

        return {'pred_y': y_pred, 'true_y': y_test}

