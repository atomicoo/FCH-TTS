#!/usr/bin/env python
"""Prepare the alignment of audio with a Text2Mel teacher model."""
__author__ = 'Atomicoo'

import sys
import os
import os.path as osp
import argparse
import time
from torch.utils.data import dataloader
from tqdm import tqdm
import numpy as np

import torch

from utils.hparams import HParam
from utils.utils import get_last_chkpt_path
from utils.transform import MinMaxNorm, StandardNorm
from utils.functional import get_fertilities, fert2align
from models import DurationExtractor
from helpers.trainer import DurationTrainer

from datasets.data_loader import Text2MelDataLoader
from datasets.dataset import SpeechDataset

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
    parser.add_argument("--checkpoint", default=None, type=str, help="Checkpoint file path")
    parser.add_argument("--ground_truth", action='store_true', help='Ground-truth melspectrogram')
    parser.add_argument("--device", default=None, type=str, help='cuda device or cpu')
    parser.add_argument("--name", default="duration", type=str, help="Append to logdir name")
    parser.add_argument('--config', default=None, type=str, help='Config file path')
    args = parser.parse_args()


    if torch.cuda.is_available():
        index = args.device if args.device else str(0 if gm is None else gm.auto_choice())
    else:
        index = 'cpu'
    device = select_device(index)

    hparams = HParam(args.config) \
            if args.config else HParam(osp.join(osp.abspath(os.getcwd()), 'config', 'default.yaml'))

    checkpoint = args.checkpoint or get_last_chkpt_path(
        osp.join(hparams.trainer.logdir, f"{hparams.data.dataset}-{args.name}"))

    extractor = DurationTrainer(hparams, device=device).load_checkpoint(checkpoint).model
    extractor.train(False)

    dataset_root = osp.join(hparams.data.datasets_path, hparams.data.dataset_dir)
    dataset = SpeechDataset(['mels', 'mlens', 'texts', 'tlens', 'files'], dataset_root, hparams.text)
    dataloader = Text2MelDataLoader(dataset, args.batch_size, mode='whole')
    normalizer = MinMaxNorm(hparams.audio.spec_min, hparams.audio.spec_max)

    pbar = tqdm(dataloader, unit="audios", unit_scale=dataloader.batch_size, \
                disable=hparams.trainer.disable_progress_bar)
    with open(osp.join(dataset.path, 'duration.txt'), 'w', encoding='utf-8') as fw:
        for it, batch in enumerate(pbar, start=1):
            mels, mlens, texts, tlens = \
                batch['mels'], batch['mlens'].squeeze(1), batch['texts'].long(), batch['tlens'].squeeze(1)
            mels, mlens, texts, tlens = \
                mels.to(device), mlens.to(device), texts.to(device), tlens.to(device)
            
            mels = normalizer(mels)

            with torch.no_grad():
                melspecs, attns = extractor((texts, tlens, mels, True))

            if args.ground_truth:
                os.makedirs(osp.join(dataset_root, 'mels-gt'), exist_ok=True)
                melspecs = normalizer.inverse(melspecs).cpu().detach()

                for i, (melspec, length) in enumerate(zip(melspecs, mlens)):
                    fname = dataset.fnames[(it-1)*args.batch_size+i]
                    melspec = np.array(melspec[:length, :])
                    np.save(osp.join(dataset_root, 'mels-gt', '%s.npy' % fname), melspec)
            
            attns, tlens, mlens = attns.cpu().detach(), tlens.cpu().detach(), mlens.cpu().detach()

            drns = get_fertilities(attns, tlens, mlens)

            for drn in drns:
                fw.write(', '.join(str(x) for x in drn) + '\n')
