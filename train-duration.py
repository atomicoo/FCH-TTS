#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train-duration.py
@Date    :   2021/01/05, Tue
@Author  :   Atomicoo
@Version :   1.0
@Contact :   atomicoo95@gmail.com
@License :   (C)Copyright 2020-2021, ShiGroup-NLP-XMU
@Desc    :   Training alignment model.
'''

__author__ = 'Atomicoo'

import argparse
import os
import os.path as osp

import torch
from utils.hparams import HParam
from helpers.logger import Logger
from helpers.trainer import DurationTrainer

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
    parser.add_argument("--warmup_epochs", default=30, type=int, help="Warmup epochs for NoamScheduler")
    parser.add_argument("--checkpoint", default=None, type=str, help="Checkpoint file path")
    parser.add_argument("--device", default=None, type=str, help='cuda device or cpu')
    parser.add_argument("--name", default="duration", type=str, help="Append to logdir name")
    parser.add_argument("--enable_wandb", action='store_true', help="Enable wandb or not")
    parser.add_argument("--project", default="parallel-speech", type=str, help="Project for wandb")
    parser.add_argument("--entity", default="atomicoo", type=str, help="Entity for wandb")
    parser.add_argument("--config", default=None, type=str, help='Config file path')
    args = parser.parse_args()

    if torch.cuda.is_available():
        index = args.device if args.device else str(0 if gm is None else gm.auto_choice())
    else:
        index = 'cpu'
    device = select_device(index)

    hparams = HParam(args.config) \
        if args.config else HParam(osp.join(osp.abspath(os.getcwd()), "config", "default.yaml"))

    loggers = Logger(
        hparams.trainer.logdir,
        hparams.data.dataset, args.name,
        wandb_info={"project": args.project, "entity": args.entity} if args.enable_wandb else None
    )

    trainer = DurationTrainer(
        hparams=hparams,
        adam_lr=0.002,
        warmup_epochs=30,
        device=device
    )

    trainer.fit(
        batch_size=args.batch_size,
        epochs=args.epochs,
        chkpt_every=10,
        checkpoint=args.checkpoint,
        loggers=loggers
    )
