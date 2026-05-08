import os
import sys

import networkx as nx
import numpy as np
import torch
import yaml
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from torch_sparse import SparseTensor
from tqdm import tqdm
from deeprobust.graph.targeted_attack import Nettack,  SGAttack
from deeprobust.graph.global_attack import MetaApprox, Metattack, DICE
from yaml import SafeLoader

from args import get_command_line_args
from attack.attackers.pga import PGA
from attack.gcn import GCN

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from data_utils.load import load_data
from utils import init_random_state, torch_sparse_tensor_to_sparse_mx, get_project_root
from scipy.sparse import csr_matrix
from deeprobust.graph.defense import SGC
from deeprobust.graph.utils import *
root = get_project_root()
Structural_Attack_Ptb = {
    "clean":[0],
    "nettack":[5],
    "SGA":[1,5],
    "mettack":[0.05,0.2],
    "DICE":[0.1,0.4],
    "PGA":[0.05,0.2]
}


def get_GNN_Model( label_num, features_shape, args, device, type="TA"):
    if args.model_name == 'GAT':
        from GNNs import GAT as GNN
        if args.robust_gnn == 'GnnGuard':
            from GNNs import GuGAT as GNN
    elif args.model_name == 'SAGE':
        from GNNs import SAGE as GNN
    elif args.model_name == 'GCN':
        from GNNs import GCN as GNN
        if args.robust_gnn == 'ProGNN':
            from GNNs import PGCN as GNN

    topk = 3 if args.dataset == 'pubmed' else 5
    attention = True if args.robust_gnn == "GnnGuard" else False
    use_pred = type == "P"
    gnn = GNN(nfeat=args.hidden_dimension * topk if use_pred else features_shape,
              nhid=args.hidden_dimension,
              nlayers=args.num_layers,
              nclass=label_num,
              use_pred=use_pred,
              dropout=args.dropout,
              weight_decay=args.weight_decay, attention=attention, device=device)

    return gnn

def select_nodes_degree(adj, idx_test, degree=10):
    G = nx.Graph()
    row, col = adj[0],adj[1]
    edge_index = list(zip(row.tolist(), col.tolist()))
    G.add_edges_from(edge_index)
    nodes = G.degree()

    return [node[0] for node in nodes if (node[1] > degree and node[0] in idx_test)]


def generate_targeted(data_name,attack_name, perturbation, seed_num, device,args=None):
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
            # surrogate = get_GNN_Model(labels.max().item() + 1, features.shape[1], args, device=device, type="E")
            # surrogate = surrogate.to(device)
            # surrogate.fit(train_data, train_iters=args.epochs)
        else:
            print("Error: Please enter the correct attack method!")
            return

        node_list = select_nodes_degree(train_data.edge_index, idx_test)
        # node_list = train_data.target_nodes
        if data_name == 'pubmed':
            node_list = node_list[:int(len(node_list) * 0.1)]
        elif data_name == 'arxiv_2023':
            node_list = node_list[:int(len(node_list) * 0.5)]

        path = f"../attack/data/target/{attack_name}/{data_name}"
        if not os.path.exists(path):
            os.mkdir(path)

        file = f"{path}/{attack_name}_{perturbation}_adv2_adj_{seed}.pt"
        modified_adj = adj.copy()
        modified_features = features.copy()

        # 2. 将节点列表平分为两组
        half = len(node_list) // 2
        struct_nodes = node_list[:half]
        feat_nodes = node_list[half:]

        # 3. 初始化两个独立的攻击模型
        if attack_name == "SGA":
            # 结构攻击模型
            model_struct = SGAttack(surrogate, attack_structure=True, attack_features=False, device=device)
            # 特征攻击模型
            model_feat = SGAttack(surrogate, attack_structure=False, attack_features=True, device=device)
        elif attack_name == "nettack":
            model_struct = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=True,
                                   device=device)
            model_feat = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=False, attack_features=True,
                                 device=device)
        else:
            raise ValueError("Error: Please enter the correct attack method!")

        for target_node  in tqdm(node_list):
            model_struct = model_struct.to(device)
            model_struct.attack(features, modified_adj, labels, target_node, perturbation, direct=True, verbose=False)
            modified_adj = model_struct.modified_adj

        modified_adj = modified_adj.tocoo()
        row = torch.tensor(modified_adj.row, dtype=torch.int64)
        col = torch.tensor(modified_adj.col, dtype=torch.int64)
        modified_adj = torch.stack([row, col], dim=0)
        data = {"target_nodes": node_list, "modified_adj": modified_adj,"modified_feature":modified_features}
        torch.save(data, file)

def generate_non_target(data_name, attack_name,  ptb_rate, device,mettModel="Self",args=None):
    #In order to reduce resource consumption, the perturbation map is generated using only one seed down
    global lambda_
    init_random_state(0)
    train_data, _ = load_data(data_name, use_dgl=False, use_text=False, seed=0)
    idx_train, idx_val, idx_test = train_data.train_id, train_data.val_id, train_data.test_id
    train_data.adj_t = SparseTensor(row=train_data.edge_index[0],
                                    col=train_data.edge_index[1],
                                    sparse_sizes=(train_data.x.shape[0], train_data.x.shape[0]))
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
    elif attack_name == "PGA":
        config_file = "pga.yaml"
        attack_config = yaml.load(open(config_file), Loader=SafeLoader)[args.dataset]

        attacker = PGA(
            attack_config=attack_config, pyg_data=train_data,
            model=surrogate, device=device, dataset_name=args.dataset
        )
        attacker.attack(
            perturbations,
            dataset=args.dataset,
        )

        modified_adj, _ = attacker.get_perturbations()
    else:
        print("Error: Please enter the correct attack method!")
        return
    path = f"{root}/attack/data/non-target/{attack_name}/{data_name}/"
    if not os.path.exists(path):
        os.mkdir(path)

    file = f"{path}/{attack_name}_adj_{ptb_rate}_{args.model_name}.pt"
    torch.save(modified_adj,file)

if __name__ == '__main__':
    attack_methods = ["nettack"]
    datasets = ['cora']
    args = get_command_line_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for model_name in ["GCN"]:
        args.model_name = model_name
        for attack_method in attack_methods:
            for dataset in datasets:
                if attack_method in ["mettack", "DICE","PGA"]:
                    for ptb_rate in Structural_Attack_Ptb[attack_method]:
                        generate_non_target(dataset,attack_method,ptb_rate,device,args=args)
                elif attack_method in ["nettack","SGA"]:
                    for ptb_rate in Structural_Attack_Ptb[attack_method]:
                        generate_targeted(dataset, attack_method,perturbation=ptb_rate, device=device,seed_num=5,args=args)

