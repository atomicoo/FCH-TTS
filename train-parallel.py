import argparse
import os
import os.path as osp

import torch
from utils.hparams import HParam
from helpers.logger import Logger
from helpers.trainer import ParallelTrainer

from utils.utils import select_device
try:
    from helpers.manager import GPUManager
except ImportError as err:
    print(err); gm = None
else:
    gm = GPUManager()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
    parser.add_argument("--epochs", default=300, type=int, help="Training epochs")
    parser.add_argument("--adam_lr", default=0.002, type=int, help="Initial learning rate for adam")
    parser.add_argument("--standardize", action='store_true', help="Standardize spectrograms")
    parser.add_argument("--ground_truth", action='store_true', help='Ground-truth melspectrogram')
    parser.add_argument("--checkpoint", default=None, type=str, help="Checkpoint file path")
    parser.add_argument("--device", default=None, type=str, help='cuda device or cpu')
    parser.add_argument("--name", default="parallel", type=str, help="Append to logdir name")
    parser.add_argument("--enable_wandb", action='store_true', help="Enable wandb or not")
    parser.add_argument("--project", default="parallel-speech", type=str, help="Project for wandb")
    parser.add_argument("--entity", default="atomicoo", type=str, help="Entity for wandb")
    parser.add_argument('--config', default=None, type=str, help='Config file path')
    args = parser.parse_args()

    index = 0 if gm is None else gm.auto_choice()
    device = select_device(args.device or str(index))

    hparams = HParam(args.config) \
        if args.config else HParam(osp.join(osp.abspath(os.getcwd()), 'config', 'default.yaml'))

    loggers = Logger(
        hparams.trainer.logdir,
        hparams.data.dataset, args.name,
        wandb_info={"project": args.project, "entity": args.entity} if args.enable_wandb else None
    )

    ground_truth = True if args.ground_truth else hparams.parallel.ground_truth
    standardize = True if args.standardize else hparams.parallel.standardize
    trainer = ParallelTrainer(
        hparams=hparams,
        adam_lr=args.adam_lr,
        ground_truth=ground_truth,
        standardize=standardize,
        device=device
    )

    trainer.fit(
        batch_size=args.batch_size,
        epochs=args.epochs,
        chkpt_every=10,
        checkpoint=args.checkpoint,
        loggers=loggers
    )
