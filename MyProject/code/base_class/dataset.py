import abc
from dataclasses import dataclass

@dataclass
class dataset:
    dName: str
    dDescription: str

    def __post_init__(self):
        self.dataset_name = self.dName
        self.dataset_description = self.dDescription

        self.dataset_folder = None
        self.dataset_path = None
        self.data = None

    def dataset_info(self):
        print(f"The name of this dataset is: {self.dataset_name}")
        print(f"This is the description of the dataset: {self.dataset_description}")

    @abc.abstractmethod
    def load_data(self):
        return