import os
import sys

import torch
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from args import get_command_line_args
from data_utils.load import load_data
from utils import knowledge_augmentation, get_e5_large_embedding, get_project_root

root = get_project_root()

def get_know(data_name,attack_type,style):
    data_obj,_,text = load_data(data_name,use_dgl=False,use_text=True,use_gpt=False,attack_type=attack_type,seed=0)
    entity = torch.load(f"{root}/attack/data/KEA/{data_name}/{data_name}_clean.pt")
    if attack_type is not None:
        adv = torch.load(f"{root}/attack/KEA/{data_name}/{data_name}_{attack_type}.pt")
        for item in adv:
            entity[item["idx"]] = item["content"]
    else:
        attack_type = "clean"
    data_obj.entity = entity
    _, knowledge = knowledge_augmentation(text, data_obj.entity, strategy='separate')
    if style == "knowsep":
        data_obj.x = get_e5_large_embedding(knowledge, 'cuda', data_name, batch_size=16)
    else:
        data_obj.x = get_e5_large_embedding(text, 'cuda', data_name, batch_size=16)
    torch.save(data_obj,f"{root}/data/emb/KEA/{data_name}/{data_name}_KEA_{style}_{attack_type}.pt")
    print(f"success")


if __name__ == '__main__':
    cfg = get_command_line_args()
    get_know(cfg.dataset,cfg.text_attack,"knowsep")
    get_know(cfg.dataset,cfg.text_attack,"KE5")