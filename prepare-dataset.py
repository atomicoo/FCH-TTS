#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train-duration.py
@Date    :   2021/01/05, Tue
@Author  :   Atomicoo
@Version :   1.0
@Contact :   atomicoo95@gmail.com
@License :   (C)Copyright 2020-2021, ShiGroup-NLP-XMU
@Desc    :   Download and preprocess datasets. Supported datasets are:

                * English female: LJSpeech
                * Mandarin female: BBSpeech (BIAOBEI)
                * Tibetan female: TBSpeech (Non-public)
                * Mongolian male: MBSpeech (Mongolian Bible)
                * Korean female: KSSpeech (Kaggle Korean Single Speech)
                * Cantonese male: HKSpeech (Common Voice, Hong Kong)
                * Japanese female: JPSpeech (JSUT Speech Corpus)

'''

__author__ = 'Atomicoo'

import sys
import os
import os.path as osp
import argparse
import pandas as pd

from utils.hparams import HParam
from utils.utils import download_file
from helpers.processor import Processor
from datasets.dataset import SpeechDataset

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', default=None, type=str, help='Config file path')
parser.add_argument('--compute', action='store_true', help='Pre-compute dataset statistics')
args = parser.parse_args()

hparams = HParam(args.config) \
        if args.config else HParam(osp.join(osp.abspath(os.getcwd()), 'config', 'default.yaml'))

datasets_path = hparams.data.datasets_path
dataset_file_url = \
    f'https://open-speech-data.oss-cn-hangzhou.aliyuncs.com/{hparams.data.dataset_dir}.tar.bz2'
dataset_file_name = osp.basename(dataset_file_url)
dataset_dir = dataset_file_name[:-8]
dataset_path = osp.join(datasets_path, dataset_dir)
wavfile_path = osp.join(dataset_path, "wavs")
melspec_path = osp.join(dataset_path, "mels")


if osp.isdir(melspec_path) and False:
    print("%s dataset folder already exists" % dataset_dir)
    sys.exit(0)
else:
    dataset_file_path = osp.join(datasets_path, dataset_file_name)
    if not osp.isfile(dataset_file_path):
        download_file(dataset_file_url, dataset_file_path)
    else:
        print("'%s' already exists" % dataset_file_name)

    if not osp.isdir(wavfile_path):
        print("extracting '%s'..." % dataset_file_name)
        os.system('cd %s; tar xvjf %s' % (datasets_path, dataset_file_name))
    else:
        print("'%s' already exists" % wavfile_path)

    dataset_root = osp.join(hparams.data.datasets_path, hparams.data.dataset_dir)
    dataset = SpeechDataset([], dataset_root, hparams.text)
    processor = Processor(hparams=hparams.audio)

    # pre process/compute
    if args.compute:
        processor.precompute(dataset_path, dataset)
    else:
        processor.preprocess(dataset_path, dataset)
