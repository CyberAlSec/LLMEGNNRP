import math
import sys, os

import numpy as np
import torch
from deeprobust.graph.utils import normalize_adj_tensor

from attack.attackers.base import BaseAttack
from torch.nn.parameter import Parameter
from copy import deepcopy
import sys

sys.path.append('...')
import utils
import torch.nn.functional as F
import scipy.sparse as sp
import torch.nn as nn


class NGA(BaseAttack):
    """NGA/FGSM.

    Parameters
    ----------
    model :
        model to attack
    nnodes : int
        number of nodes in the input graph
    feature_shape : tuple
        shape of the input node features
    attack_structure : bool
        whether to attack graph structure
    attack_features : bool
        whether to attack node features
    device: str
        'cpu' or 'cuda'
"""


    def  __init__(self, model, nnodes, feature_shape=None, attack_structure=True, attack_features=False,  decay=0.9, step=0.7, device='cpu'):

        super(NGA, self).__init__(model, nnodes, attack_structure=attack_structure, attack_features=attack_features, decay=0.9, step=0.7, device=device)
        self.decay = decay # cora= 0.6
        self.step = step # cora= 0.95

        assert not self.attack_features, "not support attacking features"

        if self.attack_features:
            self.feature_changes = Parameter(torch.FloatTensor(feature_shape))
            self.feature_changes.data.fill_(0)

    def attack(self, ori_features, ori_adj, labels, idx_train, target_node, n_perturbations, verbose=False, **kwargs):
        
        """Generate perturbations on the input graph.

        Parameters
        ----------
        ori_features : scipy.sparse.csr_matrix
            Original (unperturbed) adjacency matrix
        ori_adj : scipy.sparse.csr_matrix
            Original (unperturbed) node feature matrix
        labels :
            node labels
        idx_train:
            training node indices
        target_node : int
            target node index to be attacked
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        """

        # modified_adj = ori_adj.todense()
        # modified_features = ori_features.todense()
        # modified_adj, modified_features, labels = utils.to_tensor(modified_adj, modified_features, labels, device=self.device)

        if isinstance(ori_adj, torch.Tensor):
            if ori_adj.is_sparse:
                modified_adj = ori_adj.to_dense().clone()
            else:
                modified_adj = ori_adj.clone()
        elif sp.issparse(ori_adj):
            modified_adj = torch.FloatTensor(ori_adj.toarray())
        else:
            modified_adj = torch.from_numpy(ori_adj).float()

        modified_features = ori_features.copy()

        pseudo_labels = self.surrogate.predict().detach().argmax(1)
        pseudo_labels[idx_train] = labels[idx_train]

        self.surrogate.eval()
        if verbose == True:
            print('number of pertubations: %s' % n_perturbations)

        momentum = torch.zeros_like(modified_adj).detach()

        add_num = 0
        del_num = 0
        changed = []
        modified_adj.requires_grad = True
        change_list = [[], []]
        for i in range(n_perturbations):
            adj_norm = normalize_adj_tensor(modified_adj)

            adj_norm_nes = adj_norm + self.decay * self.step * momentum
            grad = torch.zeros_like(modified_adj).to(self.device)

            output = self.surrogate(modified_features, adj_norm_nes)
            loss = F.nll_loss(output[[target_node]], pseudo_labels[[target_node]])
            loss.backward(retain_graph=True)
            # loss = loss
            grad = torch.autograd.grad(loss, modified_adj)[0].detach()
            grad = grad[target_node]

            grad_norm = torch.norm(grad, p=1)
            grad = grad / grad_norm
            grad = grad + momentum[target_node] * self.decay
            momentum[target_node] = grad.detach()
            grad_sort = torch.argsort(torch.abs(grad), descending=True)

            for k in grad_sort:
                sign_grad = grad[k].sign()
                if sign_grad > 0 and modified_adj.data[target_node][k] == 0 and k != target_node :  # add

                    modified_adj.data[target_node][k] = 1
                    modified_adj.data[k][target_node] = 1
                    change_list[1].append(int(k))
                    add_num += 1
                    changed.append(k)
                    break

                elif sign_grad < 0 and modified_adj.data[target_node][k] == 1 and k != target_node :  # del
                    modified_adj.data[target_node][k] = 0
                    modified_adj.data[k][target_node] = 0
                    change_list[0].append(int(k))
                    changed.append(k)
                    break

        modified_adj = modified_adj.detach()
        self.check_adj(modified_adj)
        self.modified_adj = modified_adj.detach()
        # self.modified_features = modified_features
        self.add_num = add_num
        self.del_num = del_num
        self.change_list = change_list
        self.pseudo_labels = pseudo_labels


def balance_perturbation_edges(model, features, adj, target_node, perturbation, onehops_dict):
    """
    Adjusts the modified adjacency matrix to balance the number of added and deleted edges
    based on the total perturbation budget.

    Args:
        model: Model object containing modified_adj, change_list, and pseudo_labels.
        features: Tensor of node features.
        adj: Original adjacency matrix (used for shape information).
        target_node: Index of the node being targeted.
        perturbation: Total budget for edge changes.
        onehops_dict: Dictionary containing the one-hop neighbors for each node.

    Returns:
        modified_adj: The updated adjacency matrix after balancing.
    """
    if sp.issparse(features):
        device = next(model.parameters()).device  # 获取模型所在的设备
        features = torch.FloatTensor(features.toarray()).to(device)
    elif isinstance(features, np.ndarray):
        device = next(model.parameters()).device
        features = torch.from_numpy(features).to(torch.float32).to(device)

    modified_adj = model.modified_adj
    change_list = model.change_list

    # change_list[1] stores additions, change_list[0] stores deletions
    add_num = len(change_list[1])
    del_num = len(change_list[0])

    # Case 1: Number of additions exceeds half of the perturbation budget
    if add_num >= (perturbation / 2.):
        # Determine target number of final additions and deletions
        add_fin_num = math.ceil(perturbation / 2)
        if perturbation % 2 == 0:
            del_fin_num = math.ceil(perturbation / 2)
        else:
            del_fin_num = int(perturbation / 2) + 1

        add_change_num = add_num - add_fin_num
        del_change_num = del_fin_num - del_num

        # 1. Revert some added edges (set back to 0.0) based on cosine similarity
        cosine_arr = {}
        for j in change_list[1]:
            cosine_arr[j] = torch.cosine_similarity(features[target_node], features[j], dim=0)

        # Sort descending to identify which additions to remove
        cosine_sort = sorted(cosine_arr.items(), key=lambda x: x[1], reverse=True)
        for i in range(add_change_num):
            node_idx = cosine_sort[i][0]
            modified_adj[target_node, node_idx] = 0.
            modified_adj[node_idx, target_node] = 0.

        # 2. Perform additional deletions (set to 0.0) from one-hop neighbors
        cosine_arr_d = {}
        for k in onehops_dict[target_node]:
            cosine_arr_d[k] = (
                torch.cosine_similarity(features[target_node], features[k], dim=0),
                int(model.pseudo_labels[k])
            )

        cosine_sort_d = sorted(cosine_arr_d.items(), key=lambda x: x[1], reverse=True)
        for i in range(del_change_num):
            node_idx = cosine_sort_d[i][0]
            modified_adj[target_node, node_idx] = 0.
            modified_adj[node_idx, target_node] = 0.

    # Case 2: Number of deletions exceeds half of the budget (+1 buffer)
    elif del_num > (perturbation / 2.) + 1:
        # Determine target number of final deletions and additions
        del_fin_num = math.ceil(perturbation / 2)
        if perturbation % 2 == 0:
            add_fin_num = math.ceil(perturbation / 2)
        else:
            add_fin_num = int(perturbation / 2) + 1

        del_change_num = del_num - del_fin_num
        add_change_num = add_fin_num - add_num

        # 1. Revert some deleted edges (restore to 1.0) based on similarity
        cosine_arr_d = {}
        for j in change_list[0]:
            cosine_arr_d[j] = torch.cosine_similarity(features[target_node], features[j], dim=0)

        # Sort ascending
        cosine_sort_d = sorted(cosine_arr_d.items(), key=lambda x: x[1])
        for i in range(del_change_num):
            node_idx = cosine_sort_d[i][0]
            modified_adj[target_node, node_idx] = 1.
            modified_adj[node_idx, target_node] = 1.

        # 2. Perform additional additions (set to 1.0) among non-neighbor nodes
        cosine_arr = {}
        num_nodes = adj.shape[0]
        for k in range(num_nodes):
            if k not in onehops_dict[target_node] and k != target_node:
                cosine_arr[k] = torch.cosine_similarity(features[target_node], features[k], dim=0)

        # Sort ascending
        cosine_sort = sorted(cosine_arr.items(), key=lambda x: x[1])
        for i in range(add_change_num):
            node_idx = cosine_sort[i][0]
            modified_adj[target_node, node_idx] = 1.
            modified_adj[node_idx, target_node] = 1.

    return modified_adj