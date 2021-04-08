#!/usr/bin/env python
"""Prepare the voice embedding of audio with a VoiceEncoder."""
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

from encoder import VoiceEncoder, preprocess_wav


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", default=None, type=str, help='cuda device or cpu')
    parser.add_argument('--config', default=None, type=str, help='Config file path')
    args = parser.parse_args()


    if torch.cuda.is_available():
        index = args.device if args.device else str(0 if gm is None else gm.auto_choice())
    else:
        index = 'cpu'
    device = select_device(index)

    hparams = HParam(args.config) \
            if args.config else HParam(osp.join(osp.abspath(os.getcwd()), 'config', 'default.yaml'))

    dataset_root = osp.join(hparams.data.datasets_path, hparams.data.dataset_dir)
    wavs_path = osp.join(dataset_root, 'wavs')
    dataset = SpeechDataset([], dataset_root, hparams.text)

    encoder = VoiceEncoder(device)

    def get_embeddings(fpath):
        wav = preprocess_wav(fpath)
        emb = encoder.embed_utterance(wav)
        return emb

    with open(osp.join(dataset.path, 'embedding.txt'), 'w', encoding='utf-8') as fw:
        for i, fname in tqdm(enumerate(dataset.fnames)):
            embed = get_embeddings(os.path.join(wavs_path, '%s.wav' % fname))
            fw.write(', '.join(str(x) for x in embed) + '\n')
