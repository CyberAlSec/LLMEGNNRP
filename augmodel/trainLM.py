import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
import pandas as pd
from args import get_command_line_args
from augmodel.LMs.lm_trainer import LMTrainer


def run(cfg):
    if cfg.surrogate:
        seeds = [0]
    else:
        seeds = range(cfg.seed_num)
    all_acc = []
    for seed in seeds:
        cfg.seed = seed
        trainer = LMTrainer(cfg)
        trainer.train()
        acc = trainer.eval_and_save()
        all_acc.append(acc)

    if len(all_acc) > 1:
        df = pd.DataFrame(all_acc)
        for k, v in df.items():
            print(f"{k}: {v.mean():.4f} ± {v.std():.4f}")



if __name__ == '__main__':
    cfg = get_command_line_args()
    run(cfg)
