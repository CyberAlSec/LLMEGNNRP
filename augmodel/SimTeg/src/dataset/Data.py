import gzip
import json
import logging
import os
import os.path as osp
import random
import shutil

import numpy as np
import pandas as pd
import torch
from ogb.io.read_graph_pyg import read_graph_pyg
from ogb.utils.url import download_url, extract_zip
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import ToSparseTensor
from transformers import AutoTokenizer

from data_utils.load import load_data
from utils import init_random_state
from ..utils import set_logging
class Data:
    def __init__(self,train_data,text,train_masks,val_masks,test_masks,idx_trains,idx_vals,idx_tests):
        self.data = train_data
        self.text = text
        self.train_masks = train_masks
        self.val_masks = val_masks
        self.test_masks = test_masks
        self.idx_trains = idx_trains
        self.idx_vals = idx_vals
        self.idx_tests = idx_tests

logger = logging.getLogger(__name__)

class OgbText():
    def __init__(self,
                 name,
                 root='data',
                 attack_type = None,
                 pre_transform=None,
                 tokenizer = "sentence-transformers/all-MiniLM-L6-v2",
                 tokenize = True,
    ):
        self.name = name
        self.attack_type = attack_type
        self.original_root = root
        self.should_tokenize = tokenize
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer,use_fast=True) if tokenize else None
        rank = int(os.getenv("RANK",-1))
        self.data,num_labels,texts = load_data(name,use_text=True,attack_type=attack_type)
        self.data.num_labels = num_labels
        self.data.text = texts
        self.slice = None
        if self.should_tokenize:
            if not osp.exists(self.tokenized_path) and rank <= 0:
                _ = self.mapping_and_tokenizing()
            self.data.input_ids, self.data.attention_mask = self.load_cached_tokens()

    def _mapping_and_tokenizing(self):
        text_encoding = self.tokenizer(
            self.data.text,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return text_encoding.input_ids, text_encoding.attention_mask

    def get_idx_split(self,seed):
        init_random_state(seed)
        data,_ = load_data(self.name,attack_type=self.attack_type,seed=seed)
        return {"train": data.train_id, "valid": data.val_id, "test": data.test_id}


    @property
    def raw_file_names(self):
        raise NotImplementedError

    @property
    def processed_file_names(self):
        return osp.join("geometric_data_processed.pt")

    @property
    def tokenized_path(self):
        # tokenized_path = self.tokenizer.name_or_path
        tokenizer_name = self.tokenizer.name_or_path.split("/")[-1]
        tokenized_path = osp.join(self.original_root, f"{self.name}-{tokenizer_name}.pt")
        return tokenized_path

    @property
    def num_classes(self):
        return len(self.data.label_names)

    def load_cached_tokens(self):
        if osp.exists(self.tokenized_path):
            logger.info("using cached tokenized data in {}".format(self.tokenized_path))
            text_encoding = torch.load(self.tokenized_path)
            return text_encoding["input_ids"], text_encoding["attention_mask"]

    def mapping_and_tokenizing(self):
        input_ids, attention_mask = self._mapping_and_tokenizing()
        torch.save({"input_ids": input_ids, "attention_mask": attention_mask}, self.tokenized_path)
        logger.info("save the tokenized data to {}".format(self.tokenized_path))
        return input_ids, attention_mask

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)

