import numpy as np
import torch
import random

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

from utils import get_project_root

root = get_project_root()



def get_raw_text_cora(use_text=False,attack_type=None, seed=0):
    data = torch.load(f'{root}/data/cora.pt')
    text = data.text

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    # split data
    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)

    if not attack_type == None:
        advdata = torch.load(f"{root}/attack/data/orig/cora/cora_{attack_type}_orig.pt")
        advnode = [adv["idx"] for adv in advdata]
        fixed_num = len(advnode)
        if fixed_num < data.num_nodes * 0.1:
            fixed_train_nodes = np.array(advnode)
            all_nodes = [item for item in node_id if item not in fixed_train_nodes]
            data.train_id = np.sort(
                np.concatenate((all_nodes[fixed_num:int(data.num_nodes * 0.1)], fixed_train_nodes), axis=0))
        else:
            fixed_train_nodes = np.array(node_id[:data.num_nodes * 0.1])
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
