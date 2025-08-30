import gc
import os
import random

import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph import utils
from deeprobust.graph.defense import GCN, SGC
from deeprobust.graph.global_attack import MetaApprox, Metattack, DICE
from deeprobust.graph.targeted_attack import Nettack, IGAttack, SGAttack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset, Dpr2Pyg
import argparse

from gensim.models.doc2vec_corpusfile import d2v_train_epoch_dm
from openai import embeddings
from sklearn.metrics import davies_bouldin_score
from torch_sparse import SparseTensor
from tqdm import tqdm
import warnings
from args import get_command_line_args
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from scipy.sparse import csr_matrix
import networkx as nx

# from data import get_dataset
# from data import get_dataset, set_api_key
from data_utils.load import load_data, load_gpt_preds, get_structual_attack
from utils import BOW, TFIDFExtractor, torch_sparse_tensor_to_sparse_mx, Word2VecEncoder, classification_margin_list, \
    Evaluator, get_project_root, init_random_state

root = get_project_root()

warnings.filterwarnings("ignore")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
Structural_Attack_Ptb = {
    "clean":[0],
    "nettack":[0,1,2,3,4,5],
    "SGA":[0,1,2,3,4,5],
    "mettack":[0.05,0.1,0.15,0.2],
    "DICE":[0,0.1,0.2,0.3,0.4]
}

Feature_Type = {
    "OGB":{"OGB": ["OGB"],},
    "TAPE":{"TAPE": ["TA","P","E", "ensemble"]},
    "KEA":{"KEA": ["KE5", "knowsep", "ensemble"]},
    "OFA":{"OFA":["OFA"]},
    "GIANT":{"GIANT":["GIANT"]},
    "SimTeg":{"SimTeg":["SimTeg"]},
    "Random":{"Random":["Random"]},
    "E5":{"E5":["e5"]},
    "ModernBert":{"ModernBert":["ModernBert"]},
    "ChatGPT":{"ChatGPT":["ChatGPT"]},
    "Linq":{"Linq":["Linq"]},
    "Bert": {"Bert": ["Bert"]},
    "LLAMA":{"LLAMA":["LLAMA"]},

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
                f"{root}/data/emb/data/emb/{feature_type}/{args.dataset}/{feature_type}_{args.dataset}_clean_embs.pt")
        else:
            features_sur = torch.load(
                f"{root}/data/emb/data/emb/{feature_type}/{args.dataset}/{feature_type}_{args.dataset}_{args.text_attack}_embs.pt")
        features_sur = torch.FloatTensor(features_sur)
    elif feature_type == 'Random':
        features_sur = torch.randn(data.x.shape[0], 1433)
        features_sur = torch.FloatTensor(features_sur)
    elif feature_type == 'e5' or feature_type == 'ModernBert':
        if args.text_attack is None:
            LM_emb_path = f"{root}/data/emb/{feature_type}/clean/{args.dataset}/{feature_type}-large-seed{seed}.emb"
        else:
            LM_emb_path = f"{root}/data/emb/{feature_type}/{args.text_attack}/{args.dataset}/{feature_type}-large-seed{seed}.emb"

        # print(f"LM_emb_path: {LM_emb_path}")
        features_sur = torch.from_numpy(np.array(
            np.memmap(LM_emb_path, mode='r',
                      dtype=np.float16,
                      shape=(data.x.shape[0], 1024)))
        ).to(torch.float32)
    elif feature_type == 'ChatGPT':
        features_sur = torch.load(f"{root}/data/emb/ChatGPT/openai_{args.dataset}_embeddings.pt")
        # features_sur = torch.FloatTensor(features_sur)
        if type(features_sur) is list:
            features_sur = torch.stack(features_sur, dim=0)
        if args.text_attack is not None:
            advText = torch.load(f"{root}/attack/data/orig/{args.dataset}_{args.text_attack}_orig.pt")
            adv_Chat = torch.load(f"{root}/data/emb/ChatGPT/openai_{args.dataset}_{args.text_attack}_embeddings.pt")
            for item, emb in zip(advText, adv_Chat):
                features_sur[item["idx"]] = emb
    elif feature_type == 'Linq':
        features_sur = torch.load(f"{root}/data/emb/Linq/Linq_{args.dataset}_clean_emb.pt")

        if args.text_attack is not None:
            advText = torch.load(f"{root}/attack/data/orig/{args.dataset}_{args.text_attack}_orig.pt")
            adv_Chat = torch.load(f"{root}/data/emb/Linq/Linq_{args.dataset}_{args.text_attack}_emb.pt")
            for item, emb in zip(advText, adv_Chat):
                features_sur[item["idx"]] = emb
    elif feature_type == 'LLAMA':
        features_sur = torch.load(f"{root}/data/emb/Llama/Llama_{args.dataset}_clean_emb.pt")
        if args.text_attack is not None:
            advText = torch.load(f"{root}/attack/data/orig/{args.dataset}_{args.text_attack}_orig.pt")
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
        elif args.model_name == 'SAGE':
            from GNNs import SAGE as GNN
        else:
            from GNNs import GCN as GNN

        topk = 3 if args.dataset == 'pubmed' else 5

        use_pred = type == "P"
        gnn = GNN(nfeat=args.hidden_dimension * topk if use_pred else features_shape,
                  nhid=args.hidden_dimension,
                  nlayers=args.num_layers,
                  nclass=label_num,
                  use_pred= use_pred,
                  dropout=args.dropout,
                  weight_decay=args.weight_decay, device=self.device)

        return gnn

    def single_test(self, data, features, type ,labels, args,num_classes, target_node=None, test_mask = None):
        gnn = self.get_GNN_Model(num_classes,features.shape[1], args, type=type)
        gnn = gnn.to(self.device)
        gnn.fit(data, train_iters=args.epochs)

        gnn.eval()
        output = gnn.predict()

        if target_node is not None:
            margin = classification_margin(output[target_node].cpu(), labels[target_node])
            acc_test = 1.0 if (output.argmax(1)[target_node] == labels[target_node]) else 0.0
        else:
            margin = classification_margin_list(output[test_mask].cpu(), labels[test_mask].cpu())
            acc_test = self.evaluator(output[test_mask], labels[test_mask])

        del gnn
        return acc_test, margin, output


    def struct_test_poison(self, features_style, perturbation):
        seeds = range(self.args.seed_num)

        print(f"perturbation:{perturbation}")

        margin_list = {k: {item: [] for item in v} for k, v in features_style.items()}
        all_acc_list = {k: {item: [] for item in v} for k, v in features_style.items()}
        for seed in seeds:
            init_random_state(seed)
            target_nodes, modified_adj = get_structual_attack(self.args.dataset,self.args.attack_method,perturbation,seed)
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
                    acc, margin, output = self.single_test(train_data, features_surr, feature, train_data.y,
                                                           self.args,num_classes=num_classes, test_mask=test_mask)
                    margin_list[feature_key][feature].append(margin)
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
                    ensemble_margin = classification_margin_list(ensemble_out[test_mask].cpu(), labels[test_mask].cpu())
                    ensemble_acc = self.evaluator(ensemble_out[test_mask], labels[test_mask])
                    margin_list[feature_key]['ensemble'].append(ensemble_margin)
                    all_acc_list[feature_key]['ensemble'].append(ensemble_acc)

        for feature_key, feature_values in features_style.items():
            for feature in feature_values:
                acc = all_acc_list[feature_key][feature]
                margin = margin_list[feature_key][feature]
                print(
                    f"feature:{feature}, acc:{acc}, result:{round(100 * np.mean(acc),2)} ± {round(100*np.std(acc, ddof=1),2)}, margin:{round(100*np.mean(margin),2)} ± {round(100*np.std(margin, ddof=1),2)}")

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

if __name__ == '__main__':

    args = get_command_line_args()
    feature_style = Feature_Type[args.feature_type]
    # set_api_key()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    att = AttackModel(device=device, args=args)
    for perturbation in Structural_Attack_Ptb[args.attack_method]:
        att.struct_test_poison(feature_style,perturbation)