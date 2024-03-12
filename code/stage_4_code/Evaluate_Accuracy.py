from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from dataclasses import dataclass

@dataclass
class Evaluate_Accuracy(evaluate):
    def __post_init__(self):
        self.data = None

    def evaluate_accuracy(self):
        print('Evaluating Performance...')
        pass

    def evaluate_precision(self, all_labels, all_predicts):
        return precision_score(all_labels, all_predicts, average='weighted')

    def evaluate_f1(self, all_labels, all_predicts):
        return f1_score(all_labels, all_predicts, average='weighted')

    def evalulate_recall(self, all_labels, all_predicts):
        return recall_score(all_labels, all_predicts, average='weighted')

