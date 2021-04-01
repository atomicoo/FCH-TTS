"""
Hideyuki Tachibana, Katsuya Uenoyama, Shunsuke Aihara
Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention
https://arxiv.org/abs/1710.08969

Text2Mel Network. Based on the code of Erdene-Ochir Tuguldur.
"""
__author__ = 'Atomicoo'
__all__ = ['ParallelText2Mel']

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

# from hparams import HParams as hp
from .layers import Conv1d, ResidualBlock, FreqNorm

from utils.functional import mask, positional_encoding
from utils.transform import Pad


def expand_encodings(encodings, durations):
    """Expand phoneme encodings according to corresponding estimated durations

    Durations should be 0-masked, to prevent expanding of padded characters
    :param encodings:
    :param durations: (batch, time)
    :return:
    """
    encodings = [torch.repeat_interleave(e, d, dim=0)
                 for e, d in zip(encodings, durations.long())]

    return encodings


def expand_positional_encodings(durations, channels, repeat=False):
    """Expand positional encoding to align with phoneme durations

    Example:
        If repeat:
        phonemes a, b, c have durations 3,5,4
        The expanded encoding is
          a   a   a   b   b   b   b   b   c   c   c   c
        [e1, e2, e3, e1, e2, e3, e4, e5, e1, e2, e3, e4]

    Use Pad from transforms to get batched tensor.

    :param durations: (batch, time), 0-masked tensor
    :return: positional_encodings as list of tensors, (batch, time)
    """

    durations = durations.long()
    def rng(l): return list(range(l))

    if repeat:
        max_len = torch.max(durations)
        pe = positional_encoding(channels, max_len)
        idx = []
        for d in durations:
            idx.append(list(itertools.chain.from_iterable([rng(dd) for dd in d])))
        return [pe[i] for i in idx]
    else:
        max_len = torch.max(durations.sum(dim=-1))
        pe = positional_encoding(channels, max_len)
        return [pe[:s] for s in durations.sum(dim=-1)]


def round_and_mask(pred_durations, plen):
    pred_durations[pred_durations < 1] = 1  # we do not care about gradient outside training
    pred_durations = mask_durations(pred_durations, plen)  # the durations now expand only phonemes and not padded values
    pred_durations = torch.round(pred_durations)
    return pred_durations


def mask_durations(durations, plen):
    m = mask(durations.shape, plen, dim=-1).to(durations.device).float()
    return durations * m


def expand_enc(encodings, durations, mode=None):
    """Copy each phoneme encoding as many times as the duration predictor predicts"""
    encodings = Pad(0)(expand_encodings(encodings, durations))
    if mode:
        if mode == 'duration':
            encodings += Pad(0)(expand_positional_encodings(durations, encodings.shape[-1])).to(encodings.device)
        elif mode == 'standard':
            encodings += positional_encoding(encodings.shape[-1], encodings.shape[1]).to(encodings.device)
    return encodings


class TextEncoder(nn.Module):
    """Encodes input phonemes for the duration predictor and the decoder"""
    def __init__(self, hp):
        super(TextEncoder, self).__init__()
        self.kernel_size = hp.enc_kernel_size
        self.dilations = hp.enc_dilations

        self.prenet = nn.Sequential(
            nn.Embedding(hp.alphabet_size, hp.channels, padding_idx=0),
            Conv1d(hp.channels, hp.channels),
            eval(hp.activation)(),
        )

        self.res_blocks = nn.Sequential(*[
            ResidualBlock(hp.channels, self.kernel_size, d, n=2, norm=eval(hp.normalize), activation=eval(hp.activation))
            for d in self.dilations
        ])

        self.post_net1 = nn.Sequential(
            Conv1d(hp.channels, hp.channels),
        )

        self.post_net2 = nn.Sequential(
            eval(hp.activation)(),
            eval(hp.normalize)(hp.channels),
            Conv1d(hp.channels, hp.channels)
        )

    def forward(self, x):
        embedding = self.prenet(x)
        x = self.res_blocks(embedding)
        x = self.post_net1(x) + embedding
        return self.post_net2(x)


class SpecDecoder(nn.Module):
    """Decodes the expanded phoneme encoding into spectrograms"""
    def __init__(self, hp):
        super(SpecDecoder, self).__init__()
        self.kernel_size = hp.dec_kernel_size
        self.dilations = hp.dec_dilations

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hp.channels, self.kernel_size, d, n=2, norm=eval(hp.normalize), activation=eval(hp.activation))
            for d in self.dilations],
        )

        self.post_net1 = nn.Sequential(
            Conv1d(hp.channels, hp.channels),
        )

        self.post_net2 = nn.Sequential(
            ResidualBlock(hp.channels, self.kernel_size, 1, n=2),
            Conv1d(hp.channels, hp.out_channels),
            eval(hp.final_activation)()
        )

    def forward(self, x):
        xx = self.res_blocks(x)
        x = self.post_net1(xx) + x
        return self.post_net2(x)


class DurationPredictor(nn.Module):
    """Predicts phoneme log durations based on the encoder outputs"""
    def __init__(self, hp):
        super(DurationPredictor, self).__init__()

        self.layers = nn.Sequential(
            ResidualBlock(hp.channels, 4, 1, n=1, norm=eval(hp.normalize), activation=nn.ReLU),
            ResidualBlock(hp.channels, 3, 1, n=1, norm=eval(hp.normalize), activation=nn.ReLU),
            ResidualBlock(hp.channels, 1, 1, n=1, norm=eval(hp.normalize), activation=nn.ReLU),
            Conv1d(hp.channels, 1))

    def forward(self, x):
        """Outputs interpreted as log(durations)
        To get actual durations, do exp transformation
        :param x:
        :return:
        """
        return self.layers(x)


class VoiceEncoder(nn.Module):
    """Reference audio encoder"""
    def __init__(self, hp):
        super(VoiceEncoder, self).__init__()

        # Define the network
        self.lstm = nn.LSTM(hp.n_mel_channels, hp.channels, 3, batch_first=True)
        self.linear = nn.Linear(hp.channels, hp.speaker_dim)
        self.relu = nn.ReLU()

    def forward(self, mels):
        # Pass the input through the LSTM layers and retrieve the final hidden state of the last
        # layer. Apply a cutoff to 0 for negative values and L2 normalize the embeddings.
        _, (hidden, _) = self.lstm(mels)
        # Take only the hidden state of the last layer
        embeds_raw = self.relu(self.linear(hidden[-1]))
        # L2-normalize it
        embeds = embeds_raw / (torch.norm(embeds_raw, dim=1, keepdim=True) + 1e-5)
        return embeds


class Interpolate(nn.Module):
    """Use multihead attention to increase variability in expanded phoneme encodings
    
    Not used in the final model, but used in reported experiments.
    """
    def __init__(self, hp):
        super(Interpolate, self).__init__()

        ch = hp.channels
        self.att = nn.MultiheadAttention(ch, num_heads=4)
        self.norm = FreqNorm(ch)
        self.conv = Conv1d(ch, ch, kernel_size=1)

    def forward(self, x):
        xx = x.permute(1, 0, 2)  # (batch, time, channels) -> (time, batch, channels)
        xx = self.att(xx, xx, xx)[0].permute(1, 0, 2)  # (batch, time, channels)
        xx = self.conv(xx)
        return self.norm(xx) + x


class ParallelText2Mel(nn.Module):
    def __init__(self, hp):
        """Text to melspectrogram network.
        Args:
            hp: hyper parameters
        Input:
            L: (B, N) text inputs
        Outputs:
            Y: (B, T, f) predicted melspectrograms
        """
        super(ParallelText2Mel, self).__init__()
        self.hparams = hp
        self.encoder = TextEncoder(hp)
        self.decoder = SpecDecoder(hp)
        self.duration_predictor = DurationPredictor(hp)

    def forward(self, inputs):
        texts, tlens, durations, alpha = inputs
        alpha = alpha or 1.0

        encodings = self.encoder(texts)  # batch, time, channels
        prd_durans = self.duration_predictor(encodings.detach() if self.hparams.separate_duration_grad 
                                   else encodings)[..., 0]  # batch, time

        # use exp(log(durations)) = durations
        if durations is None:
            prd_durans = (round_and_mask(torch.exp(prd_durans), tlens) * alpha).long()
            encodings = expand_enc(encodings, prd_durans, mode='duration')
        else:
            encodings = expand_enc(encodings, durations, mode='duration')

        melspecs = self.decoder(encodings)
        return melspecs, prd_durans
