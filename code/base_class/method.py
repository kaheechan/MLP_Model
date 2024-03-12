import abc
from dataclasses import dataclass

class method:
    def __init__(self, mName, mDescription):
        self.method_name = mName
        self.method_description = mDescription

        self.data = None
        self.method_start_time = None
        self.method_stop_time = None
        self.method_running_time = None
        self.method_training_time = None
        self.method_testing_time = None

    @abc.abstractmethod
    def run_data(self, X_train, y_train, X_test, y_test):
        return