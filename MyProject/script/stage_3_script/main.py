from code.stage_3_code.Dataset_Loader import Dataset_Loader
from code.stage_3_code.Method_CNN import Method_CNN
import torch as th

if __name__ == '__main__':
    # ORL
    # dataset_loader_ORL = Dataset_Loader('ORL', '')
    # dataset_loader_ORL.dataset_folder = '../../data/stage_3_data'
    # dataset_loader_ORL.dataset_file = 'ORL'
    #
    # train_loader, validation_loader, test_loader = dataset_loader_ORL.load_data()
    #
    # device = th.device("cuda" if th.cuda.is_available() else "cpu")
    # method_cnn = Method_CNN('ORL', '').to(device)
    #
    # method_cnn.train_data(train_loader, validation_loader, test_loader, device)

    # MNIST
    # dataset_loader_MINST = Dataset_Loader('MNIST', '')
    # dataset_loader_MINST.dataset_folder = '../../data/stage_3_data'
    # dataset_loader_MINST.dataset_file = 'MNIST'
    #
    # train_loader, validation_loader, test_loader = dataset_loader_MINST.load_data()
    #
    # device = th.device("cuda" if th.cuda.is_available() else "cpu")
    # method_cnn = Method_CNN('MNIST', '').to(device)
    #
    # method_cnn.train_data(train_loader, validation_loader, test_loader, device)

    # CIFAR
    dataset_loader_MINST = Dataset_Loader('CIFAR', '')
    dataset_loader_MINST.dataset_folder = '../../data/stage_3_data'
    dataset_loader_MINST.dataset_file = 'CIFAR'

    train_loader, validation_loader, test_loader = dataset_loader_MINST.load_data()

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    method_cnn = Method_CNN('CIFAR', '').to(device)

    method_cnn.train_data(train_loader, validation_loader, test_loader, device)