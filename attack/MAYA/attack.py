import warnings

import nltk
import pandas as pd
import os

import torch
from MG.rl import Agent
from args import get_command_line_args
from attack.MAYA.paraphrase_models.style_transfer_paraphrase.utils import get_result
from data_utils.load import load_data
from models.models import VictimBertForSequenceClassification, VictimAutoForSequenceClassification
from MG.multi_granularity import MGAttacker
from MG.paraphrase_methods import T5, GPT2Paraphraser, BackTranslation, RoundTripParaphraser
from MG.substitute_methods import SubstituteWithBert
from tqdm import tqdm
from utils import init_random_state, get_project_root, get_model_name

# nltk.download('wordnet')
warnings.filterwarnings("ignore")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import sys
root = get_project_root()

sys.setrecursionlimit(100000)
def attack():
    if os.path.exists(dest_path):
        attack_samples = torch.load(dest_path)
        count = attack_samples['index'].values[-1] + 1
    else:
        attack_samples = pd.DataFrame(columns=['index', 'ori', 'adv', 'substitution', 'paraphrase', 'query'])
        count = 0

    total = len(samples)
    print(total)
    print(count)
    # show the information about attacking
    progress_bar = tqdm(range(count, total))
    suc = len(attack_samples.values)
    fail = count - suc
    attack_num = count
    suc_rate = suc / attack_num * 100 if attack_num != 0 else 0

    for i in progress_bar:
        progress_bar.set_description(
            '\033[0;31mparaphase:{} suc:{}  fail:{} total:{}  suc_rate:{:.2f}%\033[0m'.format(
                paraphrases, suc, fail, attack_num, suc_rate))

        info = attacker.attack(samples[i], labels[i])

        if info['adv'] is None:
            fail += 1
            attack_num += 1
            suc_rate = suc / attack_num * 100

        else:
            suc += 1
            attack_num += 1
            suc_rate = suc / attack_num * 100
            ori = samples[i][0] if isinstance(samples[i], list) else samples[i]
            new_row = pd.DataFrame([{'index': i,
                                                    'ori': ori,
                                                    'adv': info['adv'],
                                                    'substitution': info['substitution'],
                                                    'paraphrase': info['paraphrase'],
                                                    'query': info['query']}])
            attack_samples = pd.concat([attack_samples,new_row],ignore_index=True)

        torch.save(attack_samples,dest_path)

if __name__ == '__main__':
    args = get_command_line_args()
    # fill your own dataset path
    print("load dataset")
    import numpy as np
    import random

    model_name, _ = get_model_name(args.feature_type)
    name = model_name.split("/")[-1]
    ckpt_path = f"{root}/data/surr/{args.dataset}/{name}-seed0.ckpt"
    init_random_state(seed=0)
    data,num_classes,text = load_data(args.dataset,use_text=True,attack_type=args.text_attack,seed=0)
    samples = [text[i] for i in data.train_id]
    # samples = dataset[['sentence1', 'sentence2']].values
    labels = [data.y[i] for i in data.train_id]

    # fill your own victim model path
    print("load victim model")
    victim_model = VictimAutoForSequenceClassification("intfloat/e5-large",num_classes,ckpt_path)

    # choose your paraphrase models
    paraphrases = ['t5']
    paraphrase_list = []

    if 'bt' in paraphrases:
        # use back translation
        print("initialize back tran")
        paraphrase_list.append(BackTranslation())

    if 't5' in paraphrases:
        # use T5
        print("initialize T5")
        # paraphrase_list.append(T5('cuda:0'))
        paraphrase_list.append(RoundTripParaphraser(model_name="facebook/nllb-200-distilled-600M",device='cuda:0'))
    if 'gpt2' in paraphrases:
        # use gpt2 paraphrase model
        # fill your own gpt2 model path
        print("initialize gpt2")
        gpt2_path = 'paraphrase_models/style_transfer_paraphrase/paraphraser_gpt2_large'
        paraphrase_list.append(GPT2Paraphraser(gpt2_path, 'cuda:0'))

    # choose your paraphrase models
    print("initialize substitution model")
    substitution = SubstituteWithBert(victim_model,model_name='bert-base-uncased',device='cuda:0')

    # output result
    dest_path = f'{root}/data/orig/MAYA_{args.dataset}_orig.pt'

    attack_times = 5

    print("initialize attacker")
    # if use rl
    # attack_model = BertForSequenceClassification.from_pretrained(
    #     '../models/pretrained_models/mayapi_bert_for_sst2').to('cuda:0')
    # attack_model.eval()
    # agent = Agent(attack_model)
    # attacker = RLMGAttacker(attack_times, victim_model, substitution, paraphrase_list, agent)
    
    attacker = MGAttacker(attack_times, victim_model, substitution, paraphrase_list)

    # start attack
    print("start attack")
    attack()
    get_result(dest_path,data.train_id)

