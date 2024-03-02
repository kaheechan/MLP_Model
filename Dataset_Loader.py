import glob
import re
import pickle
import string
from dataclasses import dataclass

from tqdm import tqdm as tq

from code.base_class.dataset import dataset
from nltk.corpus import stopwords
from torch.utils.data import Dataset, DataLoader, random_split
import os
import pandas as pd
import matplotlib.pyplot as pt
import torch
import nltk
import re

nltk.download('stopwords')

@dataclass
class Dataset_Processor(Dataset):
    encodings: torch.Tensor
    labels: torch.Tensor

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, row):
        encoding = self.encodings[row]
        label = self.labels[row]
        return encoding, label

@dataclass
class Dataset_Loader(dataset):
    dName: str = None
    dDescription: str = None

    def __post_init__(self):
        super().__post_init__()
        self.dataset_folder = None
        self.dataset_file = None
        self.vocabulary_set = set()
        self.vocabulary_list = list()
        self.vocabulary_dict = {
            "UNK": 0
        }

    def _read_data(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as file:
            content = file.read()

        return content

    def _clean_data(self, content):
        # Lower capitalization, remove punctuation, numbers
        content = content.lower()
        content = re.sub(f'[{string.punctuation}]', '', content)
        content = re.sub(r'\d+', '', content)

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = content.split()
        tokens = [word for word in tokens if word not in stop_words]
        text = ' '.join(tokens)

        return text

    def _vocabulary_set(self, content):
        for word in content.split():
            self.vocabulary_set.add(word)

        self.vocabulary_list = list(self.vocabulary_set)

        for index, word in enumerate(self.vocabulary_list, start=1):
            self.vocabulary_dict.update({word: index})

    def _encode_data(self, content): # Training and Testing
        encode_data = []

        for word in content.split():
            encode_data.append(self.vocabulary_dict.get(word, self.vocabulary_dict.get("<UNK>")))

        return encode_data

    def load_data(self):
        # Iterate Each
        regex_pattern = r"(\d+)_(\d+)\.txt"
        encodings = []
        ids = []
        ratings = []
        data_paths = os.path.join(self.dataset_folder, self.dataset_file)

        count = 0

        for data_path in tq(glob.glob(data_paths)):
            if count >= 1000:
                break
            match = re.search(regex_pattern, data_path)
            id = match.group(1)
            rating = match.group(2)

            read_content = self._read_data(data_path)
            cleaned_content = self._clean_data(read_content)
            self._vocabulary_set(cleaned_content)
            encode_content = self._encode_data(cleaned_content)
            encodings.append(encode_content)
            ids.append(id)
            ratings.append(rating)

            count += 1

        return pd.DataFrame({'id': ids, 'rating': ratings, 'encoding': encodings})

    def encode_test_data(self):
        regex_pattern = r"(\d+)_(\d+)\.txt"
        encodings = []
        ids = []
        ratings = []
        data_paths = os.path.join(self.dataset_folder, self.dataset_file)

        count = 0

        for data_path in tq(glob.glob(data_paths)):
            if count >= 1000:
                break
            match = re.search(regex_pattern, data_path)
            id = match.group(1)
            rating = match.group(2)

            read_content = self._read_data(data_path)
            cleaned_content = self._clean_data(read_content)

            encode_content = self._encode_data(cleaned_content)
            encodings.append(encode_content)
            ids.append(id)
            ratings.append(rating)

            count += 1

        return pd.DataFrame({'id': ids, 'rating': ratings, 'encoding': encodings})