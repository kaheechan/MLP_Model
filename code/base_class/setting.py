import abc
from dataclasses import dataclass

@dataclass
class setting:
    sName: str = None
    sDescription: str = None

    def __post_init__(self):
        self.setting_name = self.sName
        self.setting_description = self.sDescription

    def prepare_object(self, sDataset, sMethod, sResult, sEvaluate):
        # Note: Parse all the object
        self.dataset = sDataset
        self.method = sMethod
        self.result = sResult
        self.evaluate = sEvaluate

    def print_object(self):
        print('dataset:', self.dataset.dataset_name, ', method:', self.method.method_name,
              ', setting:', self.setting_name, ', result:', self.result.result_name, ', evaluation:',
              self.evaluate.evaluate_name)

    @abc.abstractmethod
    def run_object(self):
        return