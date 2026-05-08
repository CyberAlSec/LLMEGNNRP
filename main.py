import gc
import os
import random

from deeprobust.graph.defense import GCNSVD, RGCN
from matplotlib import pyplot as plt
from tqdm import tqdm

from args import get_command_line_args
import torch
from deeprobust.graph.utils import *
from torch_geometric.utils import to_dense_adj, to_scipy_sparse_matrix
import warnings
import os
import numpy as np
from GNNs import SimPGCN
from certify import certify_test_set

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
warnings.filterwarnings("ignore")

from data_utils.load import load_data, load_gpt_preds, get_structual_attack
from utils import BOW, TFIDFExtractor, torch_sparse_tensor_to_sparse_mx, Word2VecEncoder, classification_margin_list, \
    Evaluator, get_project_root, init_random_state

root = get_project_root()

warnings.filterwarnings("ignore")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
Structural_Attack_Ptb = {
    "clean":[0],
    "nettack":[0,1,5],
    "SGA":[0,1,5],
    "mettack":[0,0.05,0.2],
    "DICE":[0,0.1,0.4],
    "NAG":[0,1,5],
    "PGA":[0,0.05,0.2]
}

Feature_Type = {
    "OGB":{"OGB": ["OGB"],},
    "TAPE": {"TAPE": ["TA", "P", "E", "ensemble"]},
    "KEA":{"KEA": ["KE5", "knowsep", "ensemble"]},
    "OFA":{"OFA":["OFA"]},
    "SimTeg":{"SimTeg":["SimTeg"]},
    "Random":{"Random":["Random"]},
    "E5":{"E5":["e5"]},
    "ModernBert":{"ModernBert":["ModernBERT"]},
    "ChatGPT":{"ChatGPT":["ChatGPT"]},
    "Linq":{"Linq":["Linq"]},
    "LLAMA":{"LLAMA":["LLAMA"]},
}
threh = {
    "OGB":{
        "cora": 0.35,
        "pubmed": 0.35,
        "ogbn-products": 0.6,
        "arxiv_2023": 0.9
    },
    "TAPE":{
        "cora": 0.25,
        "pubmed": 0.95,
        "ogbn-products": 0.3,
        "arxiv_2023": 0.7
    },
    "E5":{
        "cora": 0.5,
        "pubmed": 0.85,
        "ogbn-products": 0.6,
        "arxiv_2023": 0.5
    }
}

def load_features(feature_type, data, seed, args):
    topk = 3 if args.dataset == 'pubmed' else 5
    if feature_type == 'OGB':
        # print("Loading OGB features...")
        # features_sur = data.x

        if args.text_attack is None:
            fea_data = torch.load(f"{root}/data/emb/OGB/{args.dataset}/{args.dataset}_clean_embs_{seed}.pt")
        else:
            fea_data = torch.load(
                f"{root}/data/emb/OGB/{args.dataset}/{args.dataset}_{args.text_attack}_embs_{seed}.pt")
        features_sur = fea_data
    elif feature_type == 'TA':
        # print("Loading pretrained LM features (title and abstract) ...")
        if args.text_attack is None:
            LM_emb_path = f"{root}/data/emb/TAPE/clean/{args.dataset}/deberta-base-seed{seed}.emb"
        else:
            LM_emb_path = f"{root}/data/emb/TAPE/{args.text_attack}/{args.dataset}/deberta-base-seed{seed}.emb"
        # print(f"LM_emb_path: {LM_emb_path}")
        features_sur = torch.from_numpy(np.array(
            np.memmap(LM_emb_path, mode='r',
                      dtype=np.float16,
                      shape=(data.x.shape[0], 768)))
        ).to(torch.float32)
    elif feature_type == 'E':
        # print("Loading pretrained LM features (explanations) ...")
        if args.text_attack is None:
            LM_emb_path = f"{root}/data/emb/TAPE/clean/{args.dataset}2/deberta-base-seed{seed}.emb"
        else:
            LM_emb_path = f"{root}/data/emb/TAPE/{args.text_attack}/{args.dataset}2/deberta-base-seed{seed}.emb"
        # print(f"LM_emb_path: {LM_emb_path}")
        features_sur = torch.from_numpy(np.array(
            np.memmap(LM_emb_path, mode='r',
                      dtype=np.float16,
                      shape=(data.x.shape[0], 768)))
        ).to(torch.float32)
    elif feature_type == 'KE5' or feature_type == 'knowsep':
        if args.text_attack is None:
            fea_data = torch.load(f"{root}/data/emb/KEA/{args.dataset}/{args.dataset}_KEA_{feature_type}_clean.pt")
        else:
            fea_data = torch.load(
                f"{root}/data/emb/KEA/{args.dataset}/{args.dataset}_KEA_{feature_type}_{args.text_attack}.pt")
        features_sur = torch.FloatTensor(fea_data.x)
    elif feature_type == 'P':
        # print("Loading top-k prediction features ...")
        features_sur = load_gpt_preds(args.dataset, topk, data=data, attack_type=args.text_attack)
    elif feature_type == 'SimTeg':
        if args.text_attack is None:
            features_sur = torch.load(
                f"{root}/data/emb/{feature_type}/{args.dataset}/{feature_type}_{args.dataset}_clean_embs.pt")
        else:
            features_sur = torch.load(
                f"{root}/data/emb/{feature_type}/{args.dataset}/{feature_type}_{args.dataset}_{args.text_attack}_embs.pt")
        features_sur = torch.FloatTensor(features_sur)
    elif feature_type == 'Random':
        features_sur = torch.randn(data.x.shape[0], 1433)
        features_sur = torch.FloatTensor(features_sur)
    elif feature_type == 'e5' or feature_type == 'ModernBERT':
        if args.text_attack is None:
            LM_emb_path = f"{root}/data/emb/{feature_type}/clean/{args.dataset}/{feature_type}-large-seed{seed}.emb"
        else:
            LM_emb_path = f"{root}/data/emb/{feature_type}/{args.text_attack}/{args.dataset}/{feature_type}-large-seed{seed}.emb"

        # print(f"LM_emb_path: {LM_emb_path}")
        memmap_data = np.memmap(LM_emb_path, mode='r', dtype=np.float16, shape=(data.x.shape[0], 1024))
        features_sur = torch.as_tensor(memmap_data, dtype=torch.float32)
    elif feature_type == 'ChatGPT':
        features_sur = torch.load(f"{root}/data/emb/ChatGPT/openai_{args.dataset}_embeddings.pt")
        # features_sur = torch.FloatTensor(features_sur)
        if type(features_sur) is list:
            features_sur = torch.stack(features_sur, dim=0)
        if args.text_attack is not None:
            advText = torch.load(f"{root}/attack/data/orig/{args.dataset}/{args.dataset}_{args.text_attack}_orig.pt")
            adv_Chat = torch.load(f"{root}/data/emb/ChatGPT/openai_{args.dataset}_{args.text_attack}_embeddings.pt")
            for item, emb in zip(advText, adv_Chat):
                features_sur[item["idx"]] = emb
    elif feature_type == 'Linq':
        features_sur = torch.load(f"{root}/data/emb/Linq/Linq_{args.dataset}_clean_emb.pt")

        if args.text_attack is not None:
            advText = torch.load(f"{root}/attack/data/orig/{args.dataset}/{args.dataset}_{args.text_attack}_orig.pt")
            adv_Chat = torch.load(f"{root}/data/emb/Linq/Linq_{args.dataset}_{args.text_attack}_emb.pt")
            for item, emb in zip(advText, adv_Chat):
                features_sur[item["idx"]] = emb
    elif feature_type == 'LLAMA':
        features_sur = torch.load(f"{root}/data/emb/Llama/Llama_{args.dataset}_clean_emb.pt")
        if args.text_attack is not None:
            advText = torch.load(f"{root}/attack/data/orig/{args.dataset}/{args.dataset}_{args.text_attack}_orig.pt")
            adv_Chat = torch.load(f"{root}/data/emb/Llama/Llama_{args.dataset}_{args.text_attack}_emb.pt")
            for item, emb in zip(advText, adv_Chat):
                features_sur[item["idx"]] = emb
        features_sur = features_sur.to(torch.float32)
    assert torch.isfinite(features_sur).all(), "Input has NaN or Inf!"
    return features_sur


class AttackModel():

    def __init__(self, device, args):

        self.args = args
        self.device = device
        self._evaluator = Evaluator(name=args.dataset)
        self.evaluator = lambda pred, labels: self._evaluator.eval(
            {"y_pred": pred.argmax(dim=-1, keepdim=True),
             "y_true": labels.view(-1, 1)}
        )["acc"]


    def get_GNN_Model(self,label_num,features_shape,args, type="TA"):
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
        attention = True if args.robust_gnn=="GnnGuard" else False
        use_pred = type == "P"
        gnn = GNN(nfeat=args.hidden_dimension * topk if use_pred else features_shape,
                  nhid=args.hidden_dimension,
                  nlayers=args.num_layers,
                  nclass=label_num,
                  use_pred= use_pred,
                  dropout=args.dropout,
                  weight_decay=args.weight_decay, attention= attention, device=self.device)

        return gnn

    def single_test(self, data, features, type ,labels, args, num_classes, target_node=None, test_mask = None):
        if args.robust_gnn == "SimPGCN":
            use_pred = False
            model = SimPGCN(nnodes=features.shape[0], nfeat=features.shape[1],use_pred=use_pred,nhid=args.hidden_dimension, nclass=labels.max().item() + 1,
                            device=device)
            model = model.to(device)
            edge = to_scipy_sparse_matrix(data.edge_index)
            edge = sparse_mx_to_torch_sparse_tensor(edge).cuda()
            model.fit(features.cuda(), edge.cuda(), labels.cuda(), data.train_id, data.val_id)
            model.eval()
            acc_test ,output = model.test(data.test_id)
        else:
            gnn = self.get_GNN_Model(num_classes, features.shape[1], args, type=type)
            gnn = gnn.to(self.device)
            gnn.fit(data, train_iters=args.epochs)

            gnn.eval()
            output = gnn.predict()

            if target_node is not None:
                # margin = classification_margin(output[target_node].cpu(), labels[target_node])
                acc_test = 1.0 if (output.argmax(1)[target_node] == labels[target_node]) else 0.0
            else:
                # margin = classification_margin_list(output[test_mask].cpu(), labels[test_mask].cpu())
                acc_test = self.evaluator(output[test_mask], labels[test_mask])
            del gnn
        return acc_test, output


    def struct_test_poison(self, features_style, perturbation,thre=0.0):
        seeds = range(self.args.seed_num)

        print(f"perturbation:{perturbation},threshold:{thre}")

        margin_list = {k: {item: [] for item in v} for k, v in features_style.items()}
        all_acc_list = {k: {item: [] for item in v} for k, v in features_style.items()}
        for seed in seeds:
            init_random_state(seed)
            # target_nodes =None
            target_nodes, modified_adj, modified_feature = get_structual_attack(self.args.dataset,self.args.attack_method,perturbation,seed,model_name=args.model_name)
            train_data, num_classes ,text = load_data(args.dataset, use_dgl=False, use_text=True,attack_type=args.text_attack, seed=seed)
            if modified_adj is not None and modified_adj.shape[0] != 2:
                modified_adj = modified_adj.T
            train_data.text = text
            test_mask = train_data.test_mask
            if target_nodes is not None:
                test_mask = torch.zeros(train_data.x.shape[0],dtype=torch.bool)
                test_mask[target_nodes] = True

            if perturbation > 0:
                train_data.edge_index = modified_adj

            for feature_key, feature_values in features_style.items():
                ensemble_out = None
                labels = train_data.y
                for feature in feature_values:
                    if feature == "ensemble":
                        continue

                    features_surr = load_features(feature, train_data, seed, self.args)
                    train_data.x = features_surr
                    # modified_feature = torch.from_numpy(modified_feature.toarray()).float()
                    # train_data.x = modified_feature
                    # features_surr = modified_feature
                    if  isinstance(modified_adj, tuple):
                        modified_adj = modified_adj[0]
                    # modified_adj = self.filter_edges_by_similarity_torch(features_surr.cuda(),modified_adj.cuda(),thre)
                    # train_data.edge_index = modified_adj[0]
                    acc, output = self.single_test(train_data, features_surr, feature, train_data.y,
                                                           self.args,num_classes=num_classes, test_mask=test_mask)
                    all_acc_list[feature_key][feature].append(acc)

                    gc.collect()
                    torch.cuda.empty_cache()

                    if not feature_key == 'OGB':
                        if ensemble_out is None:
                            ensemble_out = output
                        else:
                            ensemble_out += output

                if len(feature_values) > 1:
                    ensemble_out /= (len(feature_values) - 1)
                    ensemble_acc = self.evaluator(ensemble_out[test_mask], labels[test_mask])
                    all_acc_list[feature_key]['ensemble'].append(ensemble_acc)

        for feature_key, feature_values in features_style.items():
            for feature in feature_values:
                acc = all_acc_list[feature_key][feature]
                print(
                    f"feature:{feature}, acc:{acc}, result:{round(100 * np.mean(acc),2)} ± {round(100*np.std(acc, ddof=1),2)}")

    def filter_edges_by_similarity_torch(self, embeddings, edge_index, threshold,
                                         metric='cosine', keep_if_greater=True):
        _EPS = 1e-12
        emb = embeddings
        ei = edge_index
        # 规范化 edge_index 到 torch.Tensor (2,E)
        if not isinstance(ei, torch.Tensor):
            ei = torch.from_numpy(np.asarray(ei))
        if ei.dim() != 2:
            raise ValueError("edge_index must be 2-D")
        if ei.size(0) == 2:
            edge_index2 = ei.long()
        elif ei.size(1) == 2:
            edge_index2 = ei.t().long()
        else:
            raise ValueError("edge_index shape must be (2, E) or (E, 2)")

        u_idx = edge_index2[0]
        v_idx = edge_index2[1]
        u = emb[u_idx].float()  # (E, D)
        v = emb[v_idx].float()  # (E, D)
        if metric == 'cosine':
            u_norm = torch.norm(u, dim=1)
            v_norm = torch.norm(v, dim=1)
            denom = u_norm * v_norm + _EPS
            sim = torch.sum(u * v, dim=1) / denom
        elif metric == 'dot':
            sim = torch.sum(u * v, dim=1)
        elif metric == 'neg_euclidean':
            dist = torch.norm(u - v, dim=1)
            sim = -dist
        else:
            raise ValueError("Unknown metric: choose 'cosine', 'dot', or 'neg_euclidean'")

        if keep_if_greater:
            mask = sim > float(threshold)
        else:
            mask = sim >= float(threshold)

        filtered = edge_index2[:, mask]
        return filtered, sim, mask

    def clean(self):
        seeds = range(self.args.seed_num)
        acc_list = []
        margin_list = []
        for seed in seeds:
            train_data, _,text = load_data(args.dataset, use_dgl=False, use_text=True, attack_type=args.text_attack, seed=seed)
            train_data.text = text
            labels = train_data.y
            test_mask = train_data.test_mask
            ensemble_out = None
            features_surr = self.load_features(self.args.feature_type ,train_data, seed, self.args)
            train_data.x = features_surr
            gnn = self.get_GNN_Model(train_data.y, features_surr.shape[1], args, type="Random")
            gnn = gnn.to(self.device)

            gnn.fit(train_data, train_iters=args.epochs)
            # gcn.fit(features, adj, labels, idx_all, idx_test, patience=30)
            gnn.eval()
            output = gnn.predict()
            test_acc = self.evaluator(
                output[test_mask], labels[test_mask])
            margin = classification_margin_list(output[test_mask].cpu(), labels[test_mask].cpu())
            acc_list.append(test_acc)
            margin_list.append(margin)
        print(
            f"Clean OGB acc:{np.mean(acc_list)}±{np.std(acc_list, ddof=1)}, margin:{np.mean(margin_list)}±{np.std(margin_list, ddof=1)}")

    def certify_model(self, features_style, args, perturbation):
        seeds = range(self.args.seed_num)
        init_random_state(0)
        target_nodes, modified_adj,modified_feat = get_structual_attack(self.args.dataset, self.args.attack_method, perturbation,
                                                          0)
        data, num_classes, text = load_data(args.dataset, use_dgl=False, use_text=True,
                                            attack_type=args.text_attack,
                                            seed=0)

        if modified_adj is not None and modified_adj.shape[0] != 2:
            modified_adj = modified_adj.T
        if perturbation > 0:
            data.edge_index = modified_adj
        for feature_key, feature_values in features_style.items():
            labels = data.y
            for feature in feature_values:
                if feature == "ensemble":
                    continue
                features_surr = load_features(feature, data, 0, self.args)
                data.x = features_surr
                gnn = self.get_GNN_Model(num_classes, features_surr.shape[1], args)
                gnn = gnn.to(self.device)
                gnn.fit(data, train_iters=args.epochs)

                test_nodes = data.test_id
                gnn = gnn.eval()
                certify_test_set(gnn, data,num_classes,test_nodes,self.device)

if __name__ == '__main__':

    args = get_command_line_args()

    # set_api_key()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    att = AttackModel(device=device, args=args)
    feature_style = Feature_Type[args.feature_type]
    for perturbation in Structural_Attack_Ptb[args.attack_method]:
        att.struct_test_poison(feature_style,perturbation)
        # att.certify_model(feature_style, args, perturbation)