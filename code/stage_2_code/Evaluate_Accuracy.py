from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from dataclasses import dataclass

@dataclass
class Evaluate_Accuracy(evaluate):
    def __post_init__(self):
        self.data = None

    def evaluate_accuracy(self):
        print('Evaluating Performance...')
        return accuracy_score(self.data['true_y'], self.data['pred_y'])

    def evaluate_f1(self):
        return f1_score(self.data['true_y'], self.data['pred_y'], average='macro')

    def evaluate_precision(self):
        return precision_score(self.data['true_y'], self.data['pred_y'], average='macro')

    def evaluate_recall(self):
        return recall_score(self.data['true_y'], self.data['pred_y'], average='macro')










