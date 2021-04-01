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
from utils.hparams import HParam
from utils.stft import MySTFT
from utils.transform import MinMaxNorm, StandardNorm

from datasets.dataset import SpeechDataset

from scipy.ndimage.morphology import binary_dilation
import webrtcvad
import struct

from voienc import VoiceEncoder, hparams, preprocess_wav


int16_max = (2 ** 15) - 1

def trim_long_silences(wav, sampling_rate=22050,
                       vad_window_length=30, 
                       vad_moving_average_width=8,
                       vad_max_silence_length=6):
    """
    Ensures that segments without voice in the waveform remain no longer than a 
    threshold determined by the VAD parameters in params.py.

    :param wav: the raw waveform as a numpy array of floats 
    :return: the same waveform with silences trimmed away (length <= original wav length)
    """
    # Compute the voice detection window size
    samples_per_window = (vad_window_length * sampling_rate) // 1000
    
    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    
    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))
    
    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=sampling_rate))
    voice_flags = np.array(voice_flags)
    
    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width
    
    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)
    
    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)
    
    return wav[audio_mask == True]


def normalize_volume(wav, target_dBFS=-20, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    rms = np.sqrt(np.mean((wav * int16_max) ** 2))
    wave_dBFS = 20 * np.log10(rms / int16_max)
    dBFS_change = target_dBFS - wave_dBFS
    if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
        return wav
    return wav * (10 ** (dBFS_change / 20))


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

        self.voice_enc = VoiceEncoder('cpu')

    def get_spectrograms(self, fpath, norm=False):
        wav, sr = librosa.load(fpath, sr=None)

        assert sr == self.hparams.sampling_rate, \
            "sample rate mismatch. expected %d, got %d at %s" % \
            (self.hparams.sampling_rate, sr, fpath)

        if norm:
            wav = normalize_volume(wav, increase_only=True)
            wav = trim_long_silences(wav, sampling_rate=self.hparams.sampling_rate)

        if len(wav) < self.hparams.segment_length + self.hparams.pad_short:
            wav = np.pad(wav, (0, self.hparams.segment_length + self.hparams.pad_short - len(wav)), \
                    mode='constant', constant_values=0.0)
        
        wav = torch.from_numpy(wav).unsqueeze(0)
        wav = wav.clamp(-1, 1)
        mel, mag = self.stft.mel_spectrogram(wav)
        return mel, mag

    def get_embeddings(self, fpath):
        wav = preprocess_wav(fpath)
        emb = self.voice_enc.embed_utterance(wav)
        return emb

    def preprocess(self, dataset_path, speech_dataset):
        """Pre-process the given dataset."""
        print("Pre-processing ...")

        wavs_path = osp.join(dataset_path, 'wavs')
        mels_path = osp.join(dataset_path, 'mels')
        os.makedirs(mels_path, exist_ok=True)
        embs_path = osp.join(dataset_path, 'embs')
        os.makedirs(embs_path, exist_ok=True)

        for i, fname in tqdm(enumerate(speech_dataset.fnames)):
            emb = self.get_embeddings(os.path.join(wavs_path, '%s.wav' % fname))

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

            np.save(osp.join(embs_path, '%s.npy' % fname), emb)
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
