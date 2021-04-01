#!/usr/bin/env python
"""Synthetize sentences into speech."""
__author__ = 'Atomicoo'

import argparse
import os
import os.path as osp
import time
from scipy.io.wavfile import write

import torch
from utils.hparams import HParam
from utils.transform import StandardNorm
from helpers.synthesizer import Synthesizer
from melgan.generator import Generator
from datasets.dataset import TextProcessor
from models import ParallelText2Mel

from utils.utils import select_device, get_last_chkpt_path
try:
    from helpers.manager import GPUManager
except ImportError as err:
    print(err); gm = None
else:
    gm = GPUManager()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size")
    parser.add_argument("--checkpoint", default=None, type=str, help="Checkpoint file path")
    parser.add_argument("--input_texts", default=None, type=str, help="Input text file path")
    parser.add_argument("--outputs_dir", default=None, type=str, help="Output wave file directory")
    parser.add_argument("--device", default=None, help="cuda device or cpu")
    parser.add_argument("--name", default="parallel", type=str, help="Append to logdir name")
    parser.add_argument("--config", default=None, type=str, help="Config file path")
    args = parser.parse_args()

    index = 0 if gm is None else gm.auto_choice()
    device = select_device(args.device or str(index))

    hparams = HParam(args.config) \
        if args.config else HParam(osp.join(osp.abspath(os.getcwd()), "config", "default.yaml"))

    logdir = osp.join(hparams.trainer.logdir, f"%s-%s" % (hparams.data.dataset, args.name))
    checkpoint = args.checkpoint or get_last_chkpt_path(logdir)

    normalizer = StandardNorm(hparams.audio.spec_mean, hparams.audio.spec_std)
    processor = TextProcessor(hparams.text)
    text2mel = ParallelText2Mel(hparams.parallel)
    text2mel.eval()

    synthesizer = Synthesizer(
        model=text2mel,
        checkpoint=checkpoint,
        processor=processor,
        normalizer=normalizer,
        device=device
    )

    text_file = args.input_texts or hparams.synthesizer.inputs_file_path
    with open(text_file, 'r', encoding='utf-8') as fr:
        texts = fr.read().strip().split('\n')
    melspecs = synthesizer.inference(texts)

    vocoder = Generator(hparams.audio.n_mel_channels).to(device)
    vocoder.eval(inference=True)
    vocoder_checkpoint = \
        osp.join(hparams.trainer.logdir, f"{hparams.data.dataset}-melgan", hparams.melgan.checkpoint)
    vocoder.load_state_dict(torch.load(vocoder_checkpoint, map_location=device))

    waves = vocoder(melspecs).squeeze(1)

    outputs_dir = args.outputs_dir or hparams.synthesizer.outputs_dir
    os.makedirs(outputs_dir, exist_ok=True)
    for i, wav in enumerate(waves, start=1):
        wav = wav.cpu().detach().numpy()
        filename = osp.join(outputs_dir, f"{time.strftime('%Y-%m-%d')}_{i:03d}.wav")
        write(filename, hparams.audio.sampling_rate, wav)
    
