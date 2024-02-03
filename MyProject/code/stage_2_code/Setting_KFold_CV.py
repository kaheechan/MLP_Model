from code.base_class.setting import setting
from sklearn.model_selection import KFold
from dataclasses import dataclass

import numpy as np

@dataclass
class Setting_KFold_CV(setting):
    def __post_init__(self):
        self.fold_number = 3

    def run_object(self):
        pass













