import os
import ssl
import sys

import networkx as nx
import nltk
import numpy as np
import torch
from tqdm import tqdm
from datasets import Dataset
import OpenAttack
from transformers import AutoTokenizer, AutoModel

from augmodel.LMs.model import BertClassifier

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from args import get_command_line_args
from data_utils.load import load_data
from utils import Evaluator, get_project_root, get_model_name

root = get_project_root()

def load_victim_model(ckpt_path: str,
                      pretrained_model_name: str = "bert-base-cased",
                      num_labels: int = 2,
                      dropout: float = 0.3,
                      seed: int = 0) -> OpenAttack.classifiers.Classifier:

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    bert_encoder = AutoModel.from_pretrained(pretrained_model_name)

    victim_model = BertClassifier(
        model=bert_encoder,
        n_labels=num_labels,
        dropout=dropout,
        seed=seed,
    )

    state_dict = torch.load(ckpt_path, map_location="cpu")
    victim_model.load_state_dict(state_dict)
    victim_model.eval()

    embedding = victim_model.bert_encoder.embeddings.word_embeddings
    victim = OpenAttack.classifiers.TransformersClassifier(victim_model, tokenizer, embedding)

    return victim



def main(dataset,attack_name,feature_type):
    # default use E5-large
    model_name, _ = get_model_name(feature_type)
    name = model_name.split("/")[-1]
    ckpt_path = f"{root}/data/surr/{dataset}/{name}-seed0.ckpt"
    data, num_classes, text = load_data(dataset=dataset, use_text=True,attack_type=None,use_gpt=False, seed=0)
    # ----------------


    victim = load_victim_model(
        ckpt_path,
        pretrained_model_name=name,
        num_labels=num_classes,
        dropout=0.3,
        seed=0
    )

    if attack_name == "bert":
        attacker = OpenAttack.attackers.BERTAttacker(mlm_path=name)
    elif attack_name == "DWord":
        attacker = OpenAttack.attackers.DeepWordBugAttacker()
    else:
        print("Error: Please enter the correct attack method!")
        return


    target = np.sort(data.target_nodes)
    data_x = [text[i] for i in target]
    data_y = [data.y.tolist()[i] for i in target]
    data = Dataset.from_dict({
        "x": data_x,
        "y": data_y
    })


    attack_eval = OpenAttack.AttackEval(
        attacker, victim,
        metrics=[OpenAttack.metric.EditDistance(),
                 OpenAttack.metric.ModificationRate()]
    )

    adversarial_samples = []
    print("\n=== Generate Adversarial Examples ===")
    i=0
    for result in tqdm(attack_eval.ieval(data), total=len(data)):  # :contentReference[oaicite:0]{index=0}
        if result["success"] is True:
            adversarial_samples.append({
                "idx": target[i],
                "content": result["result"],
            })
        i += 1

    torch.save(adversarial_samples,f"{root}/attack/data/orig/{dataset}/{dataset}_{attack_name}_orig.pt",)

if __name__ == "__main__":
    args = get_command_line_args()
    main(args.dataset,args.text_attack)


