from dataclasses import dataclass
from code.base_class.dataset import dataset
import os
import pandas as pd

@dataclass  # Note: dataset_loader is a sub-class of dataset
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
        load_df = pd.read_csv(data_path)
        return load_df