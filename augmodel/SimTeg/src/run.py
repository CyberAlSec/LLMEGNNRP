import csv
import gc
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch_geometric.transforms as T

from .dataset import load_data_bundle
from .dataset.Data import OgbText
#from .dataset import load_data_bundle
from .trainer import get_trainer_class, LMTrainer
from .utils import dataset2foldername, Evaluator

logger = logging.getLogger(__name__)


def set_single_env(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    torch.cuda.empty_cache()
    dist.destroy_process_group()
    gc.collect()


def train(args, seed=0 , return_value="valid"):

    dataObj = OgbText(root=f"./data",name=args.dataset,attack_type=args.text_attack,tokenizer="infloat/e5-large",tokenize=True)
    data = dataObj.data
    split_idx = dataObj.get_idx_split(seed)
    args.num_labels = data.num_labels

    evaluator = Evaluator(args.dataset,1,"acc")
    # trainer
    Trainer = LMTrainer
    trainer = Trainer(args, data, split_idx, evaluator)
    acc = trainer.train(return_value=return_value,attack=args.text_attack)
    del trainer, data, split_idx, evaluator
    return acc
