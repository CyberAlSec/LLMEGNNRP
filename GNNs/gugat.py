import numpy as np
import scipy as sp
from sklearn.preprocessing import normalize
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from scipy.sparse import lil_matrix
from torch_geometric.nn import MessagePassing, GATConv
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import to_torch_sparse_tensor, softmax

from .base_model import BaseModel
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from torch.nn.parameter import Parameter

class GAT(BaseModel):

    def __init__(self, nfeat, nhid, nclass, use_pred, heads=8, output_heads=1, dropout=0.5, lr=0.01,
                 nlayers=2, with_bn=True,with_relu=True, weight_decay=5e-4, attention=True, drop=False, with_bias=True, device=None):

        super(GAT, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.use_pred = use_pred
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = int(nclass)
        self.dropout = dropout
        self.lr = lr
        self.drop = drop
        self.attention = attention
        if self.use_pred:
            self.encoder = torch.nn.Embedding(nclass + 1,nhid)
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        self.gate = Parameter(torch.rand(1)) # creat a generator between [0,1]
        # self.beta = Parameter(torch.Tensor(self.n_edge))
        self.bns = torch.nn.BatchNorm1d(nhid)
        nclass = int(nclass)

        """GAT from torch-geometric"""
        self.gc1 = GATConv(nfeat, nhid, heads=8, dropout=0.6)
        self.gc2 = GATConv(nhid*8, nclass, heads=1, concat=True, dropout=0.6)

    def forward(self, x, adj):
        if self.use_pred:
            x = self.encoder(x)
            x = torch.flatten(x, start_dim=1)
        x = x.to_dense()
        adj = to_torch_sparse_tensor(adj)
        edge_index = adj._indices()

        """GCN and GAT"""
        if self.attention:
            adj = self.att_coef(x, adj, i=0)
        x = self.gc1(x, edge_index, edge_attr=adj._values())
        x = F.relu(x)
        if self.attention:  # if attention=True, use attention mechanism
            adj_2 = self.att_coef(x, adj, i=1)
            adj_values = self.gate * adj._values() + (1 - self.gate) * adj_2._values()
        else:
            adj_values = adj._values()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index, edge_attr=adj_values)

        return F.log_softmax(x, dim=1)

    def get_embed(self, x, edge_index, edge_weight=None):
        for ii, conv in enumerate(self.convs[:-1]):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index, edge_weight)
            if self.with_bn:
                x = self.bns[ii](x)
                x = F.elu(x)
        return x

    def initialize(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()


if __name__ == "__main__":
    from deeprobust.graph.data import Dataset, Dpr2Pyg

    # from deeprobust.graph.defense import GAT
    data = Dataset(root='/tmp/', name='cora')
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    gat = GAT(nfeat=features.shape[1],
              nhid=8, heads=8,
              nclass=labels.max().item() + 1,
              dropout=0.5, device='cpu')
    gat = gat.to('cpu')
    pyg_data = Dpr2Pyg(data)
    gat.fit(pyg_data, verbose=True)  # train with earlystopping
    gat.test()
    print(gat.predict())