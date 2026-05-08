import numpy as np
import scipy as sp
from sklearn.preprocessing import normalize
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from scipy.sparse import lil_matrix
from .mygat_conv import GATConv
from .base_model import BaseModel
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
from torch.nn.parameter import Parameter

class GAT(BaseModel):

    def __init__(self, nfeat, nhid, nclass,use_pred, heads=8, output_heads=1, dropout=0.5, lr=0.01,
            nlayers=2, with_bn=True, weight_decay=5e-4,attention=False,with_bias=True, device=None):

        super(GAT, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.use_pred = use_pred
        self.attention = attention
        self.gate = Parameter(torch.rand(1))  # creat a generator between [0,1]

        if self.use_pred:
            self.encoder = torch.nn.Embedding(nclass + 1,nhid)
            
        self.convs = nn.ModuleList([])
        if with_bn:
            self.bns = nn.ModuleList([])
            self.bns.append(nn.BatchNorm1d(nhid*heads))

        self.convs.append(GATConv(
            nfeat,
            nhid,
            heads=heads,
            dropout=dropout,
            bias=with_bias))

        for i in range(nlayers-2):
            self.convs.append(GATConv(nhid*heads,
                nhid, heads=heads, dropout=dropout, bias=with_bias))
            if with_bn:
                self.bns.append(nn.BatchNorm1d(nhid*heads))

        self.convs.append(GATConv(
            nhid * heads,
            nclass,
            heads=output_heads,
            concat=False,
            dropout=dropout,
            bias=with_bias))

        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.output = None
        self.best_model = None
        self.best_output = None
        self.name = 'GAT'
        self.with_bn = with_bn

    def forward(self, x, edge_index, edge_weight=None):
        if self.use_pred:
            x = self.encoder(x)
            x = torch.flatten(x,start_dim=1)

        for ii, conv in enumerate(self.convs[:-1]):
            if self.attention:
                edge_index = self.att_coef(x,edge_index,i=0)
            x = F.dropout(x.float(), p=self.dropout, training=self.training)
            x = conv(x, edge_index, edge_weight)
            if self.with_bn:
                x = self.bns[ii](x)
                x = F.elu(x)
        if self.attention:
            edge_index2 = self.att_coef(x,edge_index,i=1)
            edge_weight = self.gate * edge_weight + (1 - self.gate) * edge_index2._values()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
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
        for conv in self.convs:
            conv.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()


    def att_coef(self, fea, edge_index, is_lil=False, i=0):
        if is_lil == False:
            edge_index = edge_index._indices()
        else:
            edge_index = edge_index.tocoo()

        n_node = fea.shape[0]
        row, col = edge_index[0].cpu().data.numpy()[:], edge_index[1].cpu().data.numpy()[:]
        # row, col = edge_index[0], edge_index[1]

        fea_copy = fea.cpu().data.numpy()
        sim_matrix = cosine_similarity(X=fea_copy, Y=fea_copy)  # try cosine similarity
        # sim_matrix = torch.from_numpy(sim_matrix)
        sim = sim_matrix[row, col]
        sim[sim<0.1] = 0

        """build a attention matrix"""
        att_dense = lil_matrix((n_node, n_node), dtype=np.float32)
        att_dense[row, col] = sim
        if att_dense[0, 0] == 1:
            att_dense = att_dense - sp.diags(att_dense.diagonal(), offsets=0, format="lil")
        # normalization, make the sum of each row is 1
        att_dense_norm = normalize(att_dense, axis=1, norm='l1')


        """add learnable dropout, make character vector"""
        if self.drop:
            character = np.vstack((att_dense_norm[row, col].A1,
                                     att_dense_norm[col, row].A1))
            character = torch.from_numpy(character.T)
            drop_score = self.drop_learn_1(character)
            drop_score = torch.sigmoid(drop_score)  # do not use softmax since we only have one element
            mm = torch.nn.Threshold(0.5, 0)
            drop_score = mm(drop_score)
            mm_2 = torch.nn.Threshold(-0.49, 1)
            drop_score = mm_2(-drop_score)
            drop_decision = drop_score.clone().requires_grad_()
            # print('rate of left edges', drop_decision.sum().data/drop_decision.shape[0])
            drop_matrix = lil_matrix((n_node, n_node), dtype=np.float32)
            drop_matrix[row, col] = drop_decision.cpu().data.numpy().squeeze(-1)
            att_dense_norm = att_dense_norm.multiply(drop_matrix.tocsr())  # update, remove the 0 edges

        if att_dense_norm[0, 0] == 0:  # add the weights of self-loop only add self-loop at the first layer
            degree = (att_dense_norm != 0).sum(1).A1
            # degree = degree.squeeze(-1).squeeze(-1)
            lam = 1 / (degree + 1) # degree +1 is to add itself
            self_weight = sp.diags(np.array(lam), offsets=0, format="lil")
            att = att_dense_norm + self_weight  # add the self loop
        else:
            att = att_dense_norm

        att_adj = edge_index
        att_edge_weight = att[row, col]
        att_edge_weight = np.exp(att_edge_weight)   # exponent, kind of softmax
        att_edge_weight = torch.tensor(np.array(att_edge_weight)[0], dtype=torch.float32).cuda()

        shape = (n_node, n_node)
        new_adj = torch.sparse.FloatTensor(att_adj, att_edge_weight, shape)
        return new_adj


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
    gat.fit(pyg_data, verbose=True) # train with earlystopping
    gat.test()
    print(gat.predict())
