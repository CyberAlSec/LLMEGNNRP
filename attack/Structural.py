import os
import sys

import networkx as nx
import numpy as np
import torch
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from tqdm import tqdm
from deeprobust.graph.targeted_attack import Nettack,  SGAttack
from deeprobust.graph.global_attack import MetaApprox, Metattack, DICE

from args import get_command_line_args

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from data_utils.load import load_data
from utils import init_random_state, torch_sparse_tensor_to_sparse_mx, get_project_root
from scipy.sparse import csr_matrix
from deeprobust.graph.defense import SGC, GCN
from deeprobust.graph.utils import *
root = get_project_root()
Structural_Attack_Ptb = {
    "clean":[0],
    "nettack":[0,1,2,3,4,5],
    "SGA":[0,1,2,3,4,5],
    "mettack":[0.05,0.1,0.15,0.2],
    "DICE":[0,0.1,0.2,0.3,0.4]
}

def select_nodes_degree(adj, idx_test, degree=10):
    G = nx.Graph()
    row, col = adj[0],adj[1]
    edge_index = list(zip(row.tolist(), col.tolist()))
    G.add_edges_from(edge_index)
    nodes = G.degree()

    return [node[0] for node in nodes if (node[1] > degree and node[0] in idx_test)]


def generate_targeted(data_name,attack_name, perturbation, seed_num, device):
    seeds = range(seed_num)
    # seeds = [0]
    for seed in seeds:
        init_random_state(seed)

        train_data, _ = load_data(data_name, use_dgl=False, use_text=False, seed=seed)
        idx_train, idx_val, idx_test = train_data.train_id, train_data.val_id, train_data.test_id

        adj, features, labels = to_scipy_sparse_matrix(train_data.edge_index), csr_matrix(
            train_data.x.numpy()), train_data.y
        # Setup Surrogate model
        if attack_name == "SGA":
            surrogate = SGC(nfeat=features.shape[1],
                            nclass=labels.max().item() + 1, K=2,
                            lr=0.01, device=device).to(device)
            pyg_data = [train_data]
            surrogate.fit(pyg_data, verbose=False)  # train with earlystopping
        elif attack_name == "nettack":
            surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=16, dropout=0,
                            with_relu=False, with_bias=False, device=device).to(device)
            surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
        else:
            print("Error: Please enter the correct attack method!")
            return

        node_list = select_nodes_degree(train_data.edge_index, idx_test)
        if data_name == 'pubmed':
            node_list = node_list[:int(len(node_list) * 0.1)]
        elif data_name == 'arxiv_2023':
            node_list = node_list[:int(len(node_list) * 0.5)]

        path = f"{root}/attack/data/target/{attack_name}/{data_name}"
        if not os.path.exists(path):
            os.mkdir(path)

        file = f"{path}/{attack_name}_{perturbation}_adj_{seed}.pt"
        modified_adj = adj.tocsr()
        for target_node in tqdm(node_list):
            if attack_name == "SGA":
                model = SGAttack(surrogate, attack_structure=True, attack_features=False, device=device)
            elif attack_name == "nettack":
                model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False,
                            device=device)
            else:
                print("Error: Please enter the correct attack method!")
                break
            model = model.to(device)
            model.attack(features, modified_adj, labels, target_node, perturbation, direct=True, verbose=False)
            modified_adj = model.modified_adj
        modified_adj = modified_adj.tocoo()
        row = torch.tensor(modified_adj.row, dtype=torch.int64)
        col = torch.tensor(modified_adj.col, dtype=torch.int64)
        modified_adj = torch.stack([row, col], dim=0)
        data = {"target_nodes": node_list, "modified_adj": modified_adj}
        torch.save(data, file)

def generate_non_target(data_name, attack_name,  ptb_rate, device,mettModel="self"):
    #In order to reduce resource consumption, the perturbation map is generated using only one seed down
    global lambda_
    init_random_state(0)
    train_data, _ = load_data(data_name, use_dgl=False, use_text=False, seed=0)
    idx_train, idx_val, idx_test = train_data.train_id, train_data.val_id, train_data.test_id
    adj, features, labels = to_scipy_sparse_matrix(train_data.edge_index), csr_matrix(
        train_data.x.numpy()), train_data.y
    perturbations = int(ptb_rate * (adj.sum() // 2))
    idx_unlabeled = np.union1d(idx_val, idx_test)
    adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)
    # Setup Surrogate model
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=256, dropout=0,
                    with_relu=False,with_bias=False, device=device).to(device)
    surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)

    # Setup Attack Model
    if 'Self' in mettModel:
        lambda_ = 0
    if 'Train' in mettModel:
        lambda_ = 1
    if 'Both' in mettModel:
        lambda_ = 0.5
    if attack_name == "mettack":
        if 'A' in mettModel:
            model = MetaApprox(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
                               attack_structure=True, attack_features=False, device=device, lambda_=lambda_)
        else:
            model = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, attack_structure=True,
                              attack_features=False, device=device, lambda_=lambda_)
        model = model.to(device)
        model.attack(features, adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=False)
        modified_adj =  torch.nonzero(model.modified_adj).T
    elif attack_name == "DICE":
        model = DICE()
        adj = adj.to_dense()
        adj = to_scipy(adj)
        model.attack(adj,labels,perturbations)
        modified_adj = model.modified_adj.toarray()
        indices = np.argwhere(modified_adj != 0).T
        modified_adj = torch.tensor(indices, dtype=torch.int64)
    else:
        print("Error: Please enter the correct attack method!")
        return
    path = f"{root}/attack/data/non-target/{attack_name}/{data_name}"
    if not os.path.exists(path):
        os.mkdir(path)

    file = f"{path}/{attack_name}_adj_{ptb_rate}.pt"
    torch.save(modified_adj,file)

if __name__ == '__main__':
    cfg = get_command_line_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if cfg.attack_method in ["mettack", "DICE"]:
        for ptb_rate in Structural_Attack_Ptb[cfg.attack_method]:
            generate_non_target(cfg.dataset,cfg.attack_method,ptb_rate,device)
    elif cfg.attack_method in ["nettack","SGA"]:
        for ptb_rate in Structural_Attack_Ptb[cfg.attack_method]:
            generate_targeted(cfg.dataset, cfg.attack_method,perturbation=ptb_rate, device=device,seed_num=cfg.seed_num)

