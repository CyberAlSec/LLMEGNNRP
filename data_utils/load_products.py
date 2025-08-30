from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch
import pandas as pd
import json
import numpy as np
import os
import time

from torch_sparse import SparseTensor

from utils import time_logger, get_project_root
import random
root = get_project_root()

FILE = 'dataset/ogbn_products_orig/ogbn-products.csv'


@time_logger
def _process():
    if os.path.isfile(FILE):
        return

    print("Processing raw text...")

    data = []
    files = ['dataset/ogbn_products/Amazon-3M.raw/trn.json',
             'dataset/ogbn_products/Amazon-3M.raw/tst.json']
    for file in files:
        with open(file) as f:
            for line in f:
                data.append(json.loads(line))

    df = pd.DataFrame(data)
    df.set_index('uid', inplace=True)

    nodeidx2asin = pd.read_csv(
        'dataset/ogbn_products/mapping/nodeidx2asin.csv.gz', compression='gzip')

    dataset = PygNodePropPredDataset(
        name='ogbn-products', transform=T.ToSparseTensor())
    graph = dataset[0]
    graph.n_id = np.arange(graph.num_nodes)
    graph.n_asin = nodeidx2asin.loc[graph.n_id]['asin'].values

    graph_df = df.loc[graph.n_asin]
    graph_df['nid'] = graph.n_id
    graph_df.reset_index(inplace=True)

    if not os.path.isdir('dataset/ogbn_products_orig'):
        os.mkdir('dataset/ogbn_products_orig')
    pd.DataFrame.to_csv(graph_df, FILE,
                        index=False, columns=['uid', 'nid', 'title', 'content'])


def get_raw_text_products(use_text=False,attack_type=None,seed=0):
    data = torch.load(f'{root}/data/ogbn-subgraph.pt')
    text = data.text
    data.y = data.y.squeeze(1)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    # split data
    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)

    if not attack_type == None:
        advdata = torch.load(f"{root}/attack/data/orig/ogbn-products/ogbn-products_{attack_type}_orig.pt")
        advnode = [adv["idx"] for adv in advdata]
        fixed_num = len(advnode)
        if fixed_num < data.num_nodes*0.1:
            fixed_train_nodes = np.array(advnode)
            all_nodes = [item for item in node_id if item not in fixed_train_nodes]
            data.train_id = np.sort(np.concatenate((all_nodes[fixed_num:int(data.num_nodes * 0.1)], fixed_train_nodes), axis=0))
        else:
            fixed_train_nodes = np.array(node_id[:data.num_nodes*0.1])
            all_nodes = [item for item in node_id if item not in fixed_train_nodes]
            data.train_id = np.sort(fixed_train_nodes)
        data.val_id = np.sort(all_nodes[int(data.num_nodes * 0.1):int(data.num_nodes * 0.2)])
        data.test_id = np.sort(all_nodes[int(data.num_nodes * 0.2):])
        data.target_nodes = fixed_train_nodes
    else:
        data.train_id = np.sort(node_id[:int(data.num_nodes * 0.1)])
        data.target_nodes = data.train_id
        data.val_id = np.sort(node_id[int(data.num_nodes * 0.1):int(data.num_nodes * 0.2)])
        data.test_id = np.sort(node_id[int(data.num_nodes * 0.2):])

    data.train_mask = torch.tensor(
        [x in data.train_id for x in range(data.num_nodes)])
    data.val_mask = torch.tensor(
        [x in data.val_id for x in range(data.num_nodes)])
    data.test_mask = torch.tensor(
        [x in data.test_id for x in range(data.num_nodes)])


    if not use_text:
        return data, None

    return data, text


if __name__ == '__main__':
    data, text = get_raw_text_products(True)
    print(data)
    print(text[0])
