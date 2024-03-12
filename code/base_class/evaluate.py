import abc

from dataclasses import dataclass

@dataclass
class evaluate:
    eName: str = None
    eDescription: str = None

    def __post_init__(self):
        self.evaluate_name = self.eName
        self.evaluate_description = self.eDescription
        self.data = None

    @abc.abstractmethod
    def evaluate_data(self):
        return

    def evaluate_accuracy(self):
        return