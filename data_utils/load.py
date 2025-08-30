import os
import json
import torch
import csv

from augmodel.SimTeg.src.utils import get_project_root

root = get_project_root()

def load_gpt_preds(dataset, topk, data=None, attack_type=None):
    preds = []
    fn = f'{root}/data/TAPE/gpt_preds/{dataset}.csv'
    # print(f"Loading topk preds from {fn}")
    with open(fn, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            inner_list = []
            for value in row:
                inner_list.append(int(value))
            preds.append(inner_list)
    if dataset == 'arxiv_2023' or dataset == 'ogbn-products':
        temp = [preds[i] for i in data.node_idx]
        preds = temp
    if attack_type != None:
        adv_pred = torch.load(f"{root}/attack/data/TAPE/{dataset}/{dataset}_{attack_type}_pred.pt")
        for item in adv_pred:
            preds[item["idx"]] = item["pred"]

    pl = torch.zeros(len(preds), topk, dtype=torch.long)
    for i, pred in enumerate(preds):
        pl[i][:len(pred)] = torch.tensor(pred[:topk], dtype=torch.long)+1
    return pl


def load_data(dataset, use_dgl=False, use_text=False, use_gpt=False, attack_type=None, seed=0):
    if dataset == 'cora':
        from data_utils.load_cora import get_raw_text_cora as get_raw_text
        num_classes = 7
    elif dataset == 'pubmed':
        from data_utils.load_pubmed import get_raw_text_pubmed as get_raw_text
        num_classes = 3
        num_classes = 40
    elif dataset == 'ogbn-products':
        from data_utils.load_products import get_raw_text_products as get_raw_text
        num_classes = 47
    elif dataset == 'arxiv_2023':
        from data_utils.load_arxiv_2023 import get_raw_text_arxiv_2023 as get_raw_text
        num_classes = 40
    else:
        exit(f'Error: Dataset {dataset} not supported')

    # for training GNN
    if not use_text:
        data, _ = get_raw_text(use_text=False,attack_type=attack_type, seed=seed)
        return data, num_classes

    # for finetuning LM
    if use_gpt:
        data, text = get_raw_text(use_text=False, seed=seed)

        folder_path = f'{root}/data/TAPE/gpt_responses/{dataset}'
        print(f"using gpt: {folder_path}")
        n = data.y.shape[0]
        text = []
        for i in range(n):
            filename = str(i) + '.json'
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                content = json_data['choices'][0]['message']['content']
                text.append(content)
        if dataset == 'arxiv_2023' or dataset == 'ogbn-products':
            temp = [text[i] for i in data.node_idx]
            text = temp
        if attack_type != None:
            adv = torch.load(f"{root}/attack/data/TAPE/{dataset}/{dataset}_{attack_type}.pt")
            for item in adv:
                text[item["idx"]] = item["content"]
    else:
        data, text = get_raw_text(use_text=True, seed=seed)
        if attack_type != None:
            adv = torch.load(f"{root}/attack/data/orig/{dataset}/{dataset}_{attack_type}_orig.pt")
            for item in adv:
                text[item["idx"]] = item["content"]
        data.text = text
    return data, num_classes, text

def get_structual_attack(dataset,method, perturbation, seed):
    file = None
    if method == 'nettack':
        ptb = 1 if perturbation == 0 else perturbation
        file = f"{root}/attack/data/target/nettack/{dataset}/nettack_{ptb}_adj_{seed}.pt"
    elif method == 'SGA':
        ptb = 1 if perturbation == 0 else perturbation
        file = f"{root}/attack/data/target/sga/{dataset}/SGA_{ptb}_adj_{seed}.pt"
    elif method == 'DICE':
        ptb = 0.1 if perturbation == 0 else perturbation
        file = f"{root}/attack/data/non-target/dice/{dataset}/DICE_adj_{ptb}.pt"
    elif method == 'mettack':
        ptb = 0.05 if perturbation == 0 else perturbation
        file = f"{root}/attack/data/non-target/mettack/{dataset}/mettack_adj_{ptb}.pt"
    else:
        return None, None

    data = torch.load(file)

    if method == "nettack" or method =="SGA":
        target_node = data["target_nodes"]
        modified_adj = data["modified_adj"]
    else:
        target_node = None
        modified_adj = data
    if modified_adj.shape[1] == 2:
        modified_adj = modified_adj.T

    return target_node,modified_adj