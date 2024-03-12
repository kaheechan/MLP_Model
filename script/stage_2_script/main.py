from code.stage_2_code.Dataset_Loader import Dataset_Loader
import matplotlib.pyplot as plt
from code.stage_2_code.Method_MLP import Method_MLP
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy


if __name__ == '__main__':
    dataset_loader_test = Dataset_Loader('stage2_test', '')
    dataset_loader_test.dataset_folder = '../../data/stage_2_data/'
    dataset_loader_test.dataset_file = 'test.csv'

    dataset_loader_train = Dataset_Loader('stage2_train', '')
    dataset_loader_train.dataset_folder = '../../data/stage_2_data/'
    dataset_loader_train.dataset_file = 'train.csv'

    test_data = dataset_loader_test.load_data()
    train_data = dataset_loader_train.load_data()

    label_column_test = test_data.columns[0]
    feature_columns_test = test_data.columns[1:]

    label_column_train = train_data.columns[0]
    feature_columns_train = train_data.columns[1:]

    y_test = test_data[str(label_column_test)]
    X_test = test_data[feature_columns_test]

    y_train = train_data[str(label_column_train)]
    X_train = train_data[feature_columns_train]

    method_mlp = Method_MLP('method stage 2', '')

    result_data = method_mlp.run_data(X_train, y_train, X_test, y_test)

    evaluate_accuracy = Evaluate_Accuracy('evaluate stage 2', '')
    evaluate_accuracy.data = result_data

    print('Accuracy is: ', evaluate_accuracy.evaluate_accuracy())
    print('Precision is: ', evaluate_accuracy.evaluate_precision())
    print('Recall is: ', evaluate_accuracy.evaluate_recall())
    print('F1 Score is: ', evaluate_accuracy.evaluate_f1())

















