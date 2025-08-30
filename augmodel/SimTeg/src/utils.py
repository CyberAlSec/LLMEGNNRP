import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from typing import List, Union

# import colorlog
import torch

import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    import torch
except ImportError:
    torch = None

### Evaluator for node property prediction
class Evaluator:
    def __init__(self, name, num_tasks, eval_metric):
        self.name = name

        self.num_tasks = num_tasks
        self.eval_metric = eval_metric


    def _parse_and_check_input(self, input_dict):
        if self.eval_metric == 'rocauc' or self.eval_metric == 'acc':
            if not 'y_true' in input_dict:
                raise RuntimeError('Missing key of y_true')
            if not 'y_pred' in input_dict:
                raise RuntimeError('Missing key of y_pred')

            y_true, y_pred = input_dict['y_true'], input_dict['y_pred']

            '''
                y_true: numpy ndarray or torch tensor of shape (num_nodes num_tasks)
                y_pred: numpy ndarray or torch tensor of shape (num_nodes num_tasks)
            '''

            # converting to torch.Tensor to numpy on cpu
            if torch is not None and isinstance(y_true, torch.Tensor):
                y_true = y_true.detach().cpu().numpy()

            if torch is not None and isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.detach().cpu().numpy()

            ## check type
            if not (isinstance(y_true, np.ndarray) and isinstance(y_true, np.ndarray)):
                raise RuntimeError('Arguments to Evaluator need to be either numpy ndarray or torch tensor')

            if not y_true.shape == y_pred.shape:
                raise RuntimeError('Shape of y_true and y_pred must be the same')

            if not y_true.ndim == 2:
                raise RuntimeError('y_true and y_pred must to 2-dim arrray, {}-dim array given'.format(y_true.ndim))

            if not y_true.shape[1] == self.num_tasks:
                raise RuntimeError('Number of tasks for {} should be {} but {} given'.format(self.name, self.num_tasks, y_true.shape[1]))

            return y_true, y_pred

        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))


    def eval(self, input_dict):

        if self.eval_metric == 'rocauc':
            y_true, y_pred = self._parse_and_check_input(input_dict)
            return self._eval_rocauc(y_true, y_pred)
        elif self.eval_metric == 'acc':
            y_true, y_pred = self._parse_and_check_input(input_dict)
            return self._eval_acc(y_true, y_pred)
        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))

    @property
    def expected_input_format(self):
        desc = '==== Expected input format of Evaluator for {}\n'.format(self.name)
        if self.eval_metric == 'rocauc':
            desc += '{\'y_true\': y_true, \'y_pred\': y_pred}\n'
            desc += '- y_true: numpy ndarray or torch tensor of shape (num_nodes num_tasks)\n'
            desc += '- y_pred: numpy ndarray or torch tensor of shape (num_nodes num_tasks)\n'
            desc += 'where y_pred stores score values (for computing ROC-AUC),\n'
            desc += 'num_task is {}, and '.format(self.num_tasks)
            desc += 'each row corresponds to one node.\n'
        elif self.eval_metric == 'acc':
            desc += '{\'y_true\': y_true, \'y_pred\': y_pred}\n'
            desc += '- y_true: numpy ndarray or torch tensor of shape (num_nodes num_tasks)\n'
            desc += '- y_pred: numpy ndarray or torch tensor of shape (num_nodes num_tasks)\n'
            desc += 'where y_pred stores predicted class label (integer),\n'
            desc += 'num_task is {}, and '.format(self.num_tasks)
            desc += 'each row corresponds to one node.\n'
        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))

        return desc

    @property
    def expected_output_format(self):
        desc = '==== Expected output format of Evaluator for {}\n'.format(self.name)
        if self.eval_metric == 'rocauc':
            desc += '{\'rocauc\': rocauc}\n'
            desc += '- rocauc (float): ROC-AUC score averaged across {} task(s)\n'.format(self.num_tasks)
        elif self.eval_metric == 'acc':
            desc += '{\'acc\': acc}\n'
            desc += '- acc (float): Accuracy score averaged across {} task(s)\n'.format(self.num_tasks)
        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))

        return desc

    def _eval_rocauc(self, y_true, y_pred):
        '''
            compute ROC-AUC and AP score averaged across tasks
        '''

        rocauc_list = []

        for i in range(y_true.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                is_labeled = y_true[:,i] == y_true[:,i]
                rocauc_list.append(roc_auc_score(y_true[is_labeled,i], y_pred[is_labeled,i]))

        if len(rocauc_list) == 0:
            raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

        return {'rocauc': sum(rocauc_list)/len(rocauc_list)}

    def _eval_acc(self, y_true, y_pred):
        acc_list = []

        for i in range(y_true.shape[1]):
            is_labeled = y_true[:,i] == y_true[:,i]
            correct = y_true[is_labeled,i] == y_pred[is_labeled,i]
            acc_list.append(float(np.sum(correct))/len(correct))

        return {'acc': sum(acc_list)/len(acc_list)}

def dict_append(d: dict, u: dict):
    for key, value in u.items():
        if key in d.keys():
            d[key].append(value)
    return d




def mkdirs_if_not_exists(path: str):
    rank = -1
    if rank <= 0:
        if not os.path.exists(path):
            os.makedirs(path)


class EmbeddingHandler:
    def __init__(self, emb_path: str):
        self.emb_path = emb_path
        rank = -1
        if rank <= 0 and not os.path.exists(self.emb_path):
            os.makedirs(self.emb_path)

    def save(self, emb: torch.Tensor, saved_name: str):
        saved_name = os.path.join(self.emb_path, saved_name)
        rank =-1
        if rank <= 0:
            torch.save(emb, saved_name)

    def load(self, saved_name: str):
        if not self.has(saved_name):
            return None
        return torch.load(os.path.join(self.emb_path, saved_name))

    def has(self, saved_name: Union[str, List[str]]):
        if isinstance(saved_name, List):
            return all([os.path.exists(os.path.join(self.emb_path, name)) for name in saved_name])
        return os.path.exists(os.path.join(self.emb_path, saved_name))


def dataset2foldername(dataset):
    assert dataset in ["ogbn-arxiv", "ogbn-products", "ogbn-papers100M"]
    name_dict = {"ogbn-arxiv": "ogbn_arxiv", "ogbn-products": "ogbn_products", "ogbn-papers100M": "ogbn_papers100M"}
    return name_dict[dataset]




class RankFilter(logging.Filter):
    def filter(self, rec):
        return True
        
def is_dist():
    return False if os.getenv("WORLD_SIZE") is None else True


def set_logging():
    root = logging.getLogger()
    # NOTE: clear the std::out handler first to avoid duplicated output
    if root.hasHandlers():
        root.handlers.clear()

    root.setLevel(logging.INFO)
    log_format = "[%(name)s %(asctime)s] %(message)s"
    color_format = "%(log_color)s" + log_format

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    # console_handler.setFormatter(colorlog.ColoredFormatter(color_format))
    console_handler.addFilter(RankFilter())
    root.addHandler(console_handler)

def get_project_root():
    current_file = os.path.abspath(__file__)

    project_root = os.path.dirname(os.path.dirname(current_file))

    while not os.path.exists(os.path.join(project_root, 'README.md')):
        parent = os.path.dirname(project_root)
        if parent == project_root:
            return None
        project_root = parent

    return project_root

if __name__ == "__main__":
    set_logging()
    logger = logging.getLogger(__name__)
    logger.info("test")

