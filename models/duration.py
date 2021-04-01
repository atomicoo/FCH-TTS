"""
Hideyuki Tachibana, Katsuya Uenoyama, Shunsuke Aihara
Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention
https://arxiv.org/abs/1710.08969

Text2Mel Network. Based on the code of Erdene-Ochir Tuguldur.
"""
__author__ = 'Atomicoo'
__all__ = ['DurationExtractor']

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ZeroPad2d

# from hparams import HParams as hp
from .layers import Conv1d, WaveResidualBlock

from utils.functional import mask, positional_encoding
from utils.functional import scaled_dot_attention


class TextEncoder(nn.Module):
    """Encodes input phonemes into keys and values"""
    
    def __init__(self, hp):
        super(TextEncoder, self).__init__()
        self.kernel_size = hp.kernel_size
        self.dilations = hp.text_enc_dilations

        self.embedding = nn.Embedding(hp.alphabet_size, hp.channels, padding_idx=0)  # padding idx mapped to zero vector
        layers = [Conv1d(hp.channels, hp.channels),
                  eval(hp.activation)()]

        layers.extend([
            WaveResidualBlock(hp.channels, hp.hidden_channels, hp.kernel_size, d, causal=False)
            for d in self.dilations
        ])

        layers.append(Conv1d(hp.channels, hp.channels))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        emb = self.embedding(x)
        keys = self.layers(emb)
        values = (keys + emb) * torch.sqrt(torch.as_tensor(0.5))  # TODO: try without this
        return keys, values


class SpecEncoder(nn.Module):
    """Encodes input spectrograms into queries"""
    def __init__(self, hp):
        super(SpecEncoder, self).__init__()
        self.kernel_size = hp.kernel_size
        self.dilations = hp.spec_enc_dilations

        layers = [Conv1d(hp.out_channels, hp.channels),
                  eval(hp.activation)()]

        layers.extend([
            WaveResidualBlock(hp.channels, hp.hidden_channels, hp.kernel_size, d, causal=True)
            for d in self.dilations
        ])

        self.layers = nn.Sequential(*layers)

    def generating(self, mode):
        """Put the module into mode for sequential generation"""
        # reset queues
        for module in self.layers.children():
            if hasattr(module, 'generating'):
                module.generating(mode)

    def forward(self, x):
        return self.layers(x)


class SpecDecoder(nn.Module):
    """Decodes result of attention layer into spectrogram"""
    def __init__(self, hp):
        super(SpecDecoder, self).__init__()
        self.kernel_size = hp.kernel_size
        self.dilations = hp.spec_dec_dilations

        layers =[
            WaveResidualBlock(hp.channels, hp.hidden_channels, hp.kernel_size, d, causal=True)
            for d in self.dilations
        ]

        layers.extend([
            Conv1d(hp.channels, hp.channels),
            eval(hp.activation)(),
            Conv1d(hp.channels, hp.channels),
            eval(hp.activation)(),
            Conv1d(hp.channels, hp.channels),
            eval(hp.activation)(),
            Conv1d(hp.channels, hp.out_channels),
            eval(hp.final_activation)()
        ])
        self.layers = nn.Sequential(*layers)

    def generating(self, mode):
        """Put the module into mode for sequential generation"""
        for module in self.layers.children():
            if hasattr(module, 'generating'):
                module.generating(mode)

    def forward(self, x):
        return self.layers(x)


class ScaledDotAttention(nn.Module):
    """Scaled dot attention with positional encoding preconditioning"""

    def __init__(self, hp):
        super(ScaledDotAttention, self).__init__()

        self.noise = hp.att_noise
        self.fc_query = Conv1d(hp.channels, hp.att_hidden_channels)
        self.fc_keys = Conv1d(hp.channels, hp.att_hidden_channels)

        # share parameters
        self.fc_keys.weight = torch.nn.Parameter(self.fc_query.weight.clone())
        self.fc_keys.bias = torch.nn.Parameter(self.fc_query.bias.clone())

        self.fc_values = Conv1d(hp.channels, hp.att_hidden_channels)
        self.fc_out = Conv1d(hp.att_hidden_channels, hp.channels)

    def forward(self, q, k, v, mask=None):
        """
        :param q: queries, (batch, time1, channels1)
        :param k: keys, (batch, time2, channels1)
        :param v: values, (batch, time2, channels2)
        :param mask: boolean mask, (batch, time1, time2)
        :return: (batch, time1, channels2), (batch, time1, time2)
        """

        noise = self.noise if self.training else 0

        alignment, weights = scaled_dot_attention(self.fc_query(q),
                                                  self.fc_keys(k),
                                                  self.fc_values(v),
                                                  mask, noise=noise)
        alignment = self.fc_out(alignment)
        return alignment, weights


class DurationExtractor(nn.Module):
    def __init__(self, hp):
        super(DurationExtractor, self).__init__()
        self.hparams = hp
        self.text_enc = TextEncoder(hp)
        self.spec_enc = SpecEncoder(hp)
        self.attention = ScaledDotAttention(hp)
        self.spec_dec = SpecDecoder(hp)

    def forward(self, inputs):
        texts, tlens, mels, pos = inputs

        mels = ZeroPad2d((0,0,1,0))(mels)[:, :-1, :]
        keys, values = self.text_enc(texts)
        queries = self.spec_enc(mels)

        msk = mask(shape=(len(keys), queries.shape[1], keys.shape[1]),
                   lengths=tlens,
                   dim=-1).to(texts.device)

        if pos:
            keys += positional_encoding(keys.shape[-1], keys.shape[1], w=6.42).to(keys.device)
            queries += positional_encoding(queries.shape[-1], queries.shape[1], w=1).to(queries.device)

        seeds, attns = self.attention(queries, keys, values, mask=msk)
        melspecs = self.spec_dec(seeds+queries)
        return melspecs, attns
