import pandas as pd
import torch
from keras_preprocessing.sequence import pad_sequences
from torch.utils.data import Dataset, DataLoader, random_split
from code.stage_4_code.Dataset_Loader import Dataset_Loader, Dataset_Processor
from code.stage_4_code.Method_RNN import Method_RNN
import torch as th
import os

if __name__ == '__main__':
    # Training Data Handling [Positive]
    dataset_loader = Dataset_Loader('text_classification', '')
    dataset_loader.dataset_folder = '../../data/stage_4_data/text_classification/train/'
    dataset_loader.dataset_file = '*/*.txt'

    df = dataset_loader.load_data()  # Returns a DataFrame

    # Testing Data Handling
    dataset_loader_test = Dataset_Loader('text_classification', '')
    dataset_loader_test.dataset_folder = '../../data/stage_4_data/text_classification/test/'
    dataset_loader_test.dataset_file = '*/*.txt'

    df_test = dataset_loader_test.load_data() # Returns a DataFrame

    df['label'] = df['rating'].astype(int) >= 7
    df_test['label'] = df_test['rating'].astype(int) >= 7

    print(df)

    # Preparing for Data Processing to RNN Model
    encodings = list(df['encoding'])
    padded_encodings = pad_sequences(encodings, padding='post')

    encodings_test = list(df_test['encoding'])
    padded_encodings_test = pad_sequences(encodings_test, padding='post')

    # Tensor Object and Processing Data
    encodings = torch.tensor(padded_encodings, dtype=torch.long)
    labels = torch.tensor(df['label'].astype(float).values, dtype=torch.float)
    labels = labels.view(-1, 1)

    encodings_test = torch.tensor(padded_encodings_test, dtype=torch.long)
    labels_test = torch.tensor(df_test['label'].astype(float).values, dtype=torch.float)
    labels_test = labels_test.view(-1, 1)

    dataset_processor = Dataset_Processor(encodings, labels)
    test_dataset = Dataset_Processor(encodings_test, labels_test)

    # Splitting dataset_processor
    total_size = len(dataset_processor)
    train_size = int(total_size * 0.8)
    validation_size = int(total_size * 0.2)

    train_dataset, validation_dataset = random_split(dataset_processor, [train_size, validation_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # Train Data
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=True) # Validation Data
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True) # Test Data

    # RNN Parameters
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    vocab_size = len(dataset_loader.vocabulary_dict)
    embedding_dim = 256
    hidden_dim = 128
    output_dim = 1

    rnn_model = Method_RNN('text_classification', '', vocab_size, embedding_dim, hidden_dim, output_dim).to(device)

    # RNN Training
    rnn_model.train_data(train_loader, validation_loader, test_loader, device)











