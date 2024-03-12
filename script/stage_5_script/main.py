from code.stage_5_code.Dataset_Loader import Dataset_Loader
from code.stage_5_code.Method_GNN import Method_GNN
from torch_geometric.data import Data
import torch as th

def run_model(dName, dDescription='na'):
    # Initialize Dataset Loader
    dataset_loader = Dataset_Loader(seed=42, dName=dName, dDescription=dDescription)
    dataset_loader.dataset_name = dName
    dataset_loader.dataset_source_folder_path = f'../../data/stage_5_data/{dName}'
    data = dataset_loader.load_data()

    # Organizing Data
    X = data['graph']['X']
    y = data['graph']['y']
    A = data['graph']['utility']

    edge_array = data['graph']['edge']
    edge_index = th.tensor(edge_array, dtype=th.long).t()
    edge_weight = A.values()

    train = data['train_test_val']['idx_train']
    test = data['train_test_val']['idx_test']
    val = data['train_test_val']['idx_val']

    data = Data(x=X, edge_index=edge_index, edge_attr=edge_weight, y=y)

    number_nodes = X.shape[0]

    train_mask = th.zeros(number_nodes, dtype=th.bool)
    test_mask = th.zeros(number_nodes, dtype=th.bool)
    val_mask = th.zeros(number_nodes, dtype=th.bool)

    train_indices = th.tensor(train)
    test_indices = th.tensor(test)
    val_indices = th.tensor(val)

    train_mask[list(train_indices)] = True
    test_mask[list(test_indices)] = True
    val_mask[list(val_indices)] = True

    data.train_mask = train_mask
    data.test_mask = test_mask
    data.val_mask = val_mask

    number_nodes, number_features = X.shape
    number_classes = len(th.unique(y))

    print(f'Number Classes: {number_classes}\n')
    print(f'Number Features: {number_features}\n')

    # Feeding the data to GCN Model, to train and test
    method_gnn = Method_GNN(dName, dDescription, number_features, number_classes)

    method_gnn.train_data(data)
    method_gnn.test_data(data)

    return f'{dName}: the model runs successfully.\n'

if __name__ == '__main__':
    dName = input('Enter the dataset you want to run: \n')

    result = run_model(dName)
    print(result)