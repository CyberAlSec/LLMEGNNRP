import numpy as np
import scipy.sparse as sp
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch_sparse import SparseTensor  # 或 torch_sparse.SparseTensor
import torch_sparse
from deeprobust.graph import utils
import torch
from scipy.sparse import lil_matrix
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.preprocessing import normalize
import torch
from torch_scatter import scatter_add


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        pass

    def fit(self, pyg_data, train_iters=1000, initialize=True, verbose=False, patience=50, **kwargs):
        if initialize:
            self.initialize()

        # self.data = pyg_data[0].to(self.device)
        self.data = pyg_data.to(self.device)
        # By default, it is trained with early stopping on validation
        self.train_with_early_stopping(train_iters, patience, verbose)

    def finetune(self, edge_index, edge_weight, feat=None, train_iters=10, verbose=True):
        if verbose:
            print(f'=== finetuning {self.name} model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        labels = self.data.y
        if feat is None:
            x = self.data.x
        else:
            x = feat
        train_mask, val_mask = self.data.train_mask, self.data.val_mask
        best_loss_val = 100
        best_acc_val = 0
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(x, edge_index, edge_weight)
            loss_train = F.nll_loss(output[train_mask], labels[train_mask])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 50 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            with torch.no_grad():
                output = self.forward(x, edge_index)
            loss_val = F.nll_loss(output[val_mask], labels[val_mask])
            acc_val = utils.accuracy(output[val_mask], labels[val_mask])

            # if best_loss_val > loss_val:
            #     best_loss_val = loss_val
            #     best_output = output
            #     weights = deepcopy(self.state_dict())

            if best_acc_val < acc_val:
                best_acc_val = acc_val
                best_output = output
                weights = deepcopy(self.state_dict())

        print('best_acc_val:', best_acc_val.item())
        self.load_state_dict(weights)
        return best_output


    def _fit_with_val(self, pyg_data, train_iters=1000, initialize=True, verbose=False, **kwargs):
        if initialize:
            self.initialize()

        # self.data = pyg_data[0].to(self.device)
        self.data = pyg_data.to(self.device)
        if verbose:
            print(f'=== training {self.name} model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        labels = self.data.y
        train_mask, val_mask = self.data.train_mask, self.data.val_mask

        x, edge_index = self.data.x, self.data.edge_index
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(x, edge_index)
            loss_train = F.nll_loss(output[train_mask+val_mask], labels[train_mask+val_mask])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 50 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

    def fit_with_val(self, pyg_data, train_iters=1000, initialize=True, patience=100, verbose=False, **kwargs):
        if initialize:
            self.initialize()

        self.data = pyg_data.to(self.device)
        self.data.train_mask = self.data.train_mask + self.data.val1_mask
        self.data.val_mask = self.data.val2_mask
        self.train_with_early_stopping(train_iters, patience, verbose)

    def train_with_early_stopping(self, train_iters, patience, verbose):
        """early stopping based on the validation loss
        """
        if verbose:
            print(f'=== training {self.name} model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        labels = self.data.y
        train_mask, val_mask = self.data.train_mask, self.data.val_mask

        early_stopping = patience
        best_loss_val = 100
        best_acc_val = 0
        best_epoch = 0

        x, edge_index = self.data.x, self.data.edge_index
        edge_index.requires_grad = False
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()

            output = self.forward(x, edge_index)

            loss_train = F.nll_loss(output[train_mask], labels[train_mask])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 50 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(x, edge_index)
            loss_val = F.nll_loss(output[val_mask], labels[val_mask])
            acc_val = utils.accuracy(output[val_mask], labels[val_mask])
            # print(acc)

            # if best_loss_val > loss_val:
            #     best_loss_val = loss_val
            #     self.output = output
            #     weights = deepcopy(self.state_dict())
            #     patience = early_stopping
            #     best_epoch = i
            # else:
            #     patience -= 1

            if best_acc_val < acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
                best_epoch = i
            else:
                patience -= 1

            if i > early_stopping and patience <= 0:
                break

        if verbose:
             # print('=== early stopping at {0}, loss_val = {1} ==='.format(best_epoch, best_loss_val) )
             print('=== early stopping at {0}, acc_val = {1} ==='.format(best_epoch, best_acc_val) )
        self.load_state_dict(weights)

    def test(self):
        """Evaluate model performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        test_mask = self.data.test_mask
        labels = self.data.y
        output = self.forward(self.data.x, self.data.edge_index)
        # output = self.output
        loss_test = F.nll_loss(output[test_mask], labels[test_mask])
        acc_test = utils.accuracy(output[test_mask], labels[test_mask])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()

    def predict(self, x=None, edge_index=None, edge_weight=None):
        """
        Returns
        -------
        torch.FloatTensor
            output (log probabilities)
        """
        self.eval()
        if x is None or edge_index is None:
            x, edge_index = self.data.x, self.data.edge_index
        return self.forward(x, edge_index)

    def _ensure_contiguousness(self,
                               x,
                               edge_idx,
                               edge_weight):
        if not x.is_sparse:
            x = x.contiguous()
        if hasattr(edge_idx, 'contiguous'):
            edge_idx = edge_idx.contiguous()
        if edge_weight is not None:
            edge_weight = edge_weight.contiguous()
        return x, edge_idx, edge_weight

    # def att_coef(self, fea, edge_index, is_lil=False, i=0):
    #     n_node = fea.shape[0]
    #     # row, col = edge_index[0].cpu().data.numpy()[:], edge_index[1].cpu().data.numpy()[:]
    #     if edge_index.is_sparse:
    #         row,col = edge_index._indices()
    #         edge_index = torch.stack((row,col),dim=0)
    #         row,col = row.cpu().numpy(),col.cpu().numpy()
    #     else:
    #         row, col = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
    #
    #     fea_copy = fea.clone().detach().cpu().numpy()
    #     sim_matrix = cosine_similarity(fea_copy, fea_copy)  # try cosine similarity
    #     # sim_matrix = torch.from_numpy(sim_matrix)
    #     sim = sim_matrix[row, col]
    #     sim[sim<0.1] = 0
    #
    #     """build a attention matrix"""
    #     att_dense = lil_matrix((n_node, n_node), dtype=np.float32)
    #     att_dense[row, col] = sim
    #     if att_dense[0, 0] == 1:
    #         att_dense = att_dense - sp.diags(att_dense.diagonal(), offsets=0, format="lil")
    #     # normalization, make the sum of each row is 1
    #     att_dense_norm = normalize(att_dense, axis=1, norm='l1')
    #
    #
    #     """add learnable dropout, make character vector"""
    #     if self.drop:
    #         character = np.vstack((att_dense_norm[row, col].A1,
    #                                  att_dense_norm[col, row].A1))
    #         character = torch.from_numpy(character.T)
    #         drop_score = self.drop_learn_1(character)
    #         drop_score = torch.sigmoid(drop_score)  # do not use softmax since we only have one element
    #         mm = torch.nn.Threshold(0.5, 0)
    #         drop_score = mm(drop_score)
    #         mm_2 = torch.nn.Threshold(-0.49, 1)
    #         drop_score = mm_2(-drop_score)
    #         drop_decision = drop_score.clone().requires_grad_()
    #         # print('rate of left edges', drop_decision.sum().data/drop_decision.shape[0])
    #         drop_matrix = lil_matrix((n_node, n_node), dtype=np.float32)
    #         drop_matrix[row, col] = drop_decision.cpu().data.numpy().squeeze(-1)
    #         att_dense_norm = att_dense_norm.multiply(drop_matrix.tocsr())  # update, remove the 0 edges
    #
    #     if att_dense_norm[0, 0] == 0:  # add the weights of self-loop only add self-loop at the first layer
    #         degree = (att_dense_norm != 0).sum(1).A1
    #         # degree = degree.squeeze(-1).squeeze(-1)
    #         lam = 1 / (degree + 1) # degree +1 is to add itself
    #         self_weight = sp.diags(np.array(lam), offsets=0, format="lil")
    #         att = att_dense_norm + self_weight  # add the self loop
    #     else:
    #         att = att_dense_norm
    #
    #     att_adj = edge_index.long()
    #     att_edge_weight = att[row, col]
    #     att_edge_weight = np.exp(att_edge_weight)   # exponent, kind of softmax
    #     att_edge_weight = torch.tensor(np.array(att_edge_weight)[0], dtype=torch.float32)
    #
    #     shape = (n_node, n_node)
    #     new_adj = torch.sparse.FloatTensor(att_adj, att_edge_weight.cuda(), shape)
    #     return new_adj



    def att_coef(self, fea, edge_index, is_lil=False, i=0):
        """
        GPU + torch_scatter 优化版 att_coef.
        预期：
          - fea: torch.Tensor, float32, device 可为 'cuda'
          - edge_index: torch.LongTensor shape [2, E], 放在同一 device 上
          - 若 self.drop 为 True, 需要 self.drop_learn_1 支持在 GPU 上运行
        返回：torch.sparse_coo_tensor, shape (n_node, n_node)
        """
        device = fea.device

        # --- 处理 edge_index 输入 ---

        if isinstance(edge_index,torch_sparse.SparseTensor):
            row,col = edge_index.storage.row(),edge_index.storage.col()
            ei = torch.stack([row,col], dim=0).to(device)
        elif edge_index.layout == torch.sparse_coo:
            ei = edge_index.indices().to(device)
        else:
            ei = edge_index.to(device)

        row, col = ei[0].long(), ei[1].long()
        E = row.shape[0]
        n_node = fea.shape[0]
        if E == 0:
            # 无边 -> 返回单位矩阵
            idx = torch.arange(n_node, device=device, dtype=torch.long)
            indices = torch.stack([idx, idx], dim=0)
            values = torch.ones(n_node, device=device, dtype=torch.float32)
            return torch.sparse_coo_tensor(indices, values, (n_node, n_node)).coalesce()

        # --- 特征归一化 (L2) 并只对边计算余弦相似度 ---
        fea = fea.float()
        norm = fea.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
        fea_norm = (fea / norm) # [N, D]
        sim = (fea_norm[row] * fea_norm[col]).sum(dim=1)  # [E]
        sim = torch.where(sim < 0.1, torch.zeros_like(sim), sim)  # threshold

        # 若全部为0 -> 直接返回带自环的单位或适当矩阵
        if (sim.abs() < 1e-12).all():
            idx = torch.arange(n_node, device=device, dtype=torch.long)
            indices = torch.stack([idx, idx], dim=0)
            values = torch.ones(n_node, device=device, dtype=torch.float32)
            return torch.sparse_coo_tensor(indices, values, (n_node, n_node)).coalesce()

        # --- 行归一化 (L1): outgoing sum = 1 ---
        row_sum = scatter_add(sim, row, dim=0, dim_size=n_node)  # [N]
        denom = row_sum[row] + 1e-12
        values = sim / denom  # normalized per row, shape [E]

        # --- 可学习的 dropout（向量化） ---
        if getattr(self, 'drop', False):
            # 建 pair_id 用于快速查找反向边 (u,v) <-> (v,u)
            pair_ids = row.long() * (n_node) + col.long()  # 确保 long，避免溢出
            rev_pair_ids = col.long() * (n_node) + row.long()

            # 排序并用 searchsorted 找到反向边索引（在 GPU 上也适用）
            sorted_pair_ids, order = torch.sort(pair_ids)
            pos = torch.searchsorted(sorted_pair_ids, rev_pair_ids)
            # 防止越界访问
            pos_clamped = pos.clone()
            pos_clamped[pos_clamped >= E] = 0

            # 判断是否真正匹配
            matched = (pos < E) & (sorted_pair_ids[pos_clamped] == rev_pair_ids)
            rev_index = torch.full((E,), -1, dtype=torch.long, device=device)
            if matched.any():
                rev_index[matched] = order[pos_clamped[matched]]

            # 取反向的 values（若不存在则为 0）
            rev_values = torch.zeros_like(values)
            valid = rev_index >= 0
            if valid.any():
                rev_values[valid] = values[rev_index[valid]]

            character = torch.stack([values, rev_values], dim=1)  # [E,2]

            # drop_learn_1 在 GPU 上运行，输出形状 [E] 或 [E,1]
            drop_score = self.drop_learn_1(character).view(-1)
            drop_score = torch.sigmoid(drop_score)
            drop_decision = (drop_score > 0.5).float()  # threshold
            values = values * drop_decision

        # --- 若没有自环则添加自环 (按原逻辑) ---
        has_self_loops = (row == col).any().item()
        if not has_self_loops:
            nonzero_mask = (values != 0).float()
            deg = scatter_add(nonzero_mask, row, dim=0, dim_size=n_node)  # 出度（非零条数）
            lam = 1.0 / (deg + 1.0)  # shape [N]
            diag_idx = torch.arange(n_node, device=device, dtype=torch.long)
            diag_indices = torch.stack([diag_idx, diag_idx], dim=0)
            diag_values = lam  # self weight

            # 合并原边与对角
            indices = torch.cat([torch.stack([row, col], dim=0), diag_indices], dim=1)
            values = torch.cat([values, diag_values], dim=0)
        else:
            indices = torch.stack([row, col], dim=0)

        # --- exponentiate (原代码) 并移除零权重保持稀疏 ---
        values = torch.exp(values)
        nz_mask = values != 0
        if nz_mask.sum().item() == 0:
            # 兜底：全部为零 -> 返回单位矩阵
            idx = torch.arange(n_node, device=device, dtype=torch.long)
            indices = torch.stack([idx, idx], dim=0)
            values = torch.ones(n_node, device=device, dtype=torch.float32)
        else:
            indices = indices[:, nz_mask]
            values = values[nz_mask]
        indices = indices.long()
        # new_adj = torch.sparse_coo_tensor(indices.long(), values.float(), (n_node, n_node))
        # return new_adj.coalesce()
        new_adj = SparseTensor(row=indices[0], col=indices[1], value=values.float(), sparse_sizes=(n_node,n_node))
        return new_adj
