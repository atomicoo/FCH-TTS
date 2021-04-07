import sys
import os
import os.path as osp
import time
import math
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F

import librosa
from pydub import AudioSegment
from utils.stft import MySTFT
from utils.transform import MinMaxNorm, StandardNorm


def read_audio_from_file(path, format=None):
    format = format or path.split('.')[-1]
    sound = AudioSegment.from_file(path, format=format)
    return sound

def match_target_amplitude(sound, target_dBFS=-20):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

def trim_long_silences(sound, silence_len=700, silence_thresh=None, padding=100):
    sound_dBFS = sound.dBFS
    sil_thresh = silence_thresh or sound_dBFS-10
    trimmed_sound = AudioSegment.strip_silence(
        sound, silence_len=silence_len, silence_thresh=sil_thresh, padding=padding)
    return trimmed_sound

def from_pydub_to_librosa(sound, sampling_rate=22050):
    samples = sound.get_array_of_samples()
    wav = np.array(samples).astype(np.float32)
    wav /= np.iinfo(samples.typecode).max
    wav = librosa.core.resample(wav, sound.frame_rate, sampling_rate, res_type='kaiser_best')
    return wav


class Processor:
    def __init__(self, hparams):
        super(Processor, self).__init__()
        self.hparams = hparams

        self.stft = MySTFT(
                filter_length=self.hparams.filter_length,
                hop_length=self.hparams.hop_length,
                win_length=self.hparams.win_length,
                n_mel_channels=self.hparams.n_mel_channels,
                sampling_rate=self.hparams.sampling_rate,
                mel_fmin=self.hparams.mel_fmin,
                mel_fmax=self.hparams.mel_fmax)

    def get_spectrograms(self, fpath, norm=False):
        # wav, sr = librosa.load(fpath, sr=None)
        sound = read_audio_from_file(fpath, format='wav')

        if self.hparams.force_frame_rate:
            assert sound.frame_rate == self.hparams.sampling_rate, \
                "sample rate mismatch. expected %d, got %d at %s" % \
                (self.hparams.sampling_rate, sound.frame_rate, fpath)

        if norm:
            sound = match_target_amplitude(sound)
            sound = trim_long_silences(sound)

        wav = from_pydub_to_librosa(sound, self.hparams.sampling_rate)

        if len(wav) < self.hparams.segment_length + self.hparams.pad_short:
            wav = np.pad(wav, (0, self.hparams.segment_length + self.hparams.pad_short - len(wav)), \
                    mode='constant', constant_values=0.0)
        
        wav = torch.from_numpy(wav).unsqueeze(0)
        wav = wav.clamp(-1, 1)
        mel, mag = self.stft.mel_spectrogram(wav)
        return mel, mag

    def preprocess(self, dataset_path, speech_dataset):
        """Pre-process the given dataset."""
        print("Pre-processing ...")

        wavs_path = osp.join(dataset_path, 'wavs')
        mels_path = osp.join(dataset_path, 'mels')
        os.makedirs(mels_path, exist_ok=True)

        for i, fname in tqdm(enumerate(speech_dataset.fnames)):
            mel, _ = self.get_spectrograms(osp.join(wavs_path, '%s.wav' % fname), self.hparams.normalize)
            mel = mel.squeeze(0)
            # mel = standard_norm(mel).clamp(hp.scale_min,hp.scale_max)

            t = mel.size(1)
            # Marginal padding for reduction shape sync.
            reduction_rate = self.hparams.reduction_rate
            num_paddings = reduction_rate - (t % reduction_rate) if t % reduction_rate != 0 else 0
            b, e = math.floor(num_paddings/2), math.ceil(num_paddings/2)
            mel = np.pad(mel, [[0, 0], [b, e]], mode="edge")
            mel = mel.transpose(1, 0)
            # Reduction
            # mel = mel[::hp.reduction_rate, :]

            np.save(osp.join(mels_path, '%s.npy' % fname), mel)


    def precompute(self, dataset_path, speech_dataset):
        """Pre-compute the given dataset."""
        print("Pre-computing ...")

        wavs_path = osp.join(dataset_path, 'wavs')
        os.makedirs(wavs_path, exist_ok=True)

        spec_min, spec_max = 0.0, 0.0
        spec_means, spec_stds = [], []
        lengths = []

        for i, fname in tqdm(enumerate(speech_dataset.fnames)):
            mel, _ = self.get_spectrograms(osp.join(wavs_path, '%s.wav' % fname), self.hparams.normalize)
            mel = mel.squeeze(0)

            lengths.append(mel.size(1))

            spec_min = mel.min().item() if mel.min()<spec_min else spec_min
            spec_max = mel.max().item() if mel.max()>spec_max else spec_max
            spec_means.append(mel.mean().item())
            spec_stds.append(mel.std().item())

        lengths = torch.tensor(lengths)

        spec_means = torch.tensor(spec_means)
        spec_stds = torch.tensor(spec_stds)
        spec_mean = ((spec_means * lengths).sum() / lengths.sum()).item()
        spec_std = (torch.sqrt((spec_stds.pow(2) * (lengths-1)).sum() / (lengths-1).sum())).item()

        print("spec_min: {:.5f}, spec_max: {:.5f}".format(spec_min, spec_max))
        print("spec_mean: {:.5f}, spec_std: {:.5f}".format(spec_mean, spec_std))
