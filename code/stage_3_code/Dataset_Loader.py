import pickle
from dataclasses import dataclass
from code.base_class.dataset import dataset
from torch.utils.data import TensorDataset, DataLoader, random_split
import os
import pandas as pd
import matplotlib.pyplot as pt
import torch

@dataclass
class Dataset_Loader(dataset):
    dName: str = None
    dDescription: str = None

    def __post_init__(self):
        super().__post_init__()
        self.dataset_folder = None
        self.dataset_file = None

    def load_data(self):
        data_path = os.path.join(self.dataset_folder, self.dataset_file)
        print(f"Loading data for {data_path}...")

        # Open Files
        file = open(data_path, 'rb')
        data = pickle.load(file)
        file.close()

        # Train Data, Test Data
        train_image = torch.tensor([instance['image'] for instance in data['train']]).float()
        train_label = torch.tensor([instance['label'] for instance in data['train']])
        test_image = torch.tensor([instance['image'] for instance in data['test']]).float()
        test_label = torch.tensor([instance['label'] for instance in data['test']])

        #
        if self.dName == 'ORL':
            train_image = train_image.permute(0, 3, 1, 2)
            test_image = test_image.permute(0, 3, 1, 2)

            train_label -= 1
            test_label -= 1

        elif self.dName == 'MNIST':
            train_image = train_image.unsqueeze(1)
            test_image = test_image.unsqueeze(1)

        #
        elif self.dName == 'CIFAR':
            train_image = train_image.permute(0, 3, 1, 2)
            test_image = test_image.permute(0, 3, 1, 2)

        # Tensor Dataset
        train_dataset = TensorDataset(train_image, train_label)
        test_dataset = TensorDataset(test_image, test_label)

        # Validation Split
        validation_split_ratio = 0.2
        train_split_ratio = 0.8

        validation_size = int(len(train_dataset) * validation_split_ratio)
        train_size = int(len(train_dataset) * train_split_ratio)

        train_dataset, validation_dataset = random_split(train_dataset, [train_size, validation_size])

        # Data Loader
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        return train_loader, validation_loader, test_loader



