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
from vocoder.models import MelGANGenerator
from utils.audio import dynamic_range_decompression
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
    parser.add_argument("--melgan_checkpoint", default=None, type=str, help="Checkpoint file path of melgan")
    parser.add_argument("--input_texts", default=None, type=str, help="Input text file path")
    parser.add_argument("--outputs_dir", default=None, type=str, help="Output wave file directory")
    parser.add_argument("--device", default=None, help="cuda device or cpu")
    parser.add_argument("--name", default="parallel", type=str, help="Append to logdir name")
    parser.add_argument("--config", default=None, type=str, help="Config file path")
    args = parser.parse_args()

    if torch.cuda.is_available():
        index = args.device if args.device else str(0 if gm is None else gm.auto_choice())
    else:
        index = 'cpu'
    device = select_device(index)

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

    print('Synthesizing...')
    since = time.time()
    text_file = args.input_texts or hparams.synthesizer.inputs_file_path
    with open(text_file, 'r', encoding='utf-8') as fr:
        texts = fr.read().strip().split('\n')
    melspecs = synthesizer.inference(texts)
    print(f"Inference {len(texts)} spectrograms, total elapsed {time.time()-since:.3f}s. Done.")

    vocoder_checkpoint = args.melgan_checkpoint or \
        osp.join(hparams.trainer.logdir, f"{hparams.data.dataset}-melgan", hparams.melgan.checkpoint)
    ckpt = torch.load(vocoder_checkpoint, map_location=device)

    decompressed  = dynamic_range_decompression(melspecs)
    decompressed_log10 = torch.log10(decompressed)
    mu = torch.tensor(ckpt['stats']['mu']).unsqueeze(1)
    var = torch.tensor(ckpt['stats']['var']).unsqueeze(1)
    sigma = torch.sqrt(var)
    decompressed_log10_norm = (decompressed_log10 - mu) / sigma
    melspecs = (decompressed_log10 - mu) / sigma

    vocoder = MelGANGenerator(**ckpt['config']).to(device)
    vocoder.remove_weight_norm()
    vocoder.load_state_dict(ckpt['model'])

    waves = vocoder(melspecs).squeeze(1)
    print(f"Generate {len(texts)} audios, total elapsed {time.time()-since:.3f}s. Done.")

    print('Saving audio...')
    outputs_dir = args.outputs_dir or hparams.synthesizer.outputs_dir
    os.makedirs(outputs_dir, exist_ok=True)
    for i, wav in enumerate(waves, start=1):
        wav = wav.cpu().detach().numpy()
        filename = osp.join(outputs_dir, f"{time.strftime('%Y-%m-%d')}_{i:03d}.wav")
        write(filename, hparams.audio.sampling_rate, wav)
    print(f"Audios saved to {outputs_dir}. Done.")

    print(f'Done. ({time.time()-since:.3f}s)')
    
