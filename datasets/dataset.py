"""Dataset class for speech dataset."""
import os
import os.path as osp
import re
import codecs
import unicodedata
import numpy as np
from g2p_en import G2p

import torch
from torch.utils.data import Dataset


class TextProcessor:

    # phonemes = [
    #     'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
    #     'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2',
    #     'B', 'CH', 'D', 'DH',
    #     'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2',
    #     'F', 'G', 'HH',
    #     'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2',
    #     'JH', 'K', 'L', 'M', 'N', 'NG',
    #     'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2',
    #     'P', 'R', 'S', 'SH', 'T', 'TH',
    #     'UH0', 'UH1', 'UH2', 'UW', 'UW0', 'UW1', 'UW2',
    #     'V', 'W', 'Y', 'Z', 'ZH'
    # ]
    g2p = G2p()

    def __init__(self, hparams):
        self.units = self.graphemes = hparams.graphemes
        self.phonemes = hparams.phonemes
        self.phonemize = hparams.use_phonemes
        if self.phonemize:
            self.units = self.phonemes
        self.specials = hparams.specials
        self.punctuations = hparams.punctuations
        self.units = self.specials + self.units + self.punctuations
        self.txt2idx = {txt: idx for idx, txt in enumerate(self.units)}
        self.idx2txt = {idx: txt for idx, txt in enumerate(self.units)}

    def normalize(self, text):
        text = text.lower()
        text = re.sub("[ ]+", " ", text)
        # keep_re = "[^" + str(self.graphemes+self.punctuations) +"]"
        # text = re.sub(keep_re, " ", text)  # remove
        text = [ch if ch in self.graphemes+self.punctuations else ' ' for ch in text]
        text = list(text)
        if self.phonemize:
            text = self.g2p(''.join(text))
        return text

    def __call__(self, texts, max_n=None):
        if not isinstance(texts, (str, list)):
            raise TypeError("Inputs must be str or list(str)")
        if isinstance(texts, str):
            texts = [texts]
        normalized_texts = [self.normalize(line) for line in texts]  # text normalization
        tlens = [len(l) for l in normalized_texts]
        max_n = max_n or max(tlens)
        texts = np.zeros((len(normalized_texts), max_n), np.long)
        for i, text in enumerate(normalized_texts):
            texts[i, :len(text)] = [self.txt2idx.get(ch, 1) for ch in text]
        return texts, tlens


class SpeechDataset(Dataset):
    def __init__(self, keys, root, hparams):
        self.keys = keys
        self.path = root
        self.text_proc = TextProcessor(hparams)
        self.fnames, self.text_lengths, self.texts = \
            self.read_metadata(osp.join(self.path, 'metadata.csv'))
        del self.text_proc
        self.embeds = self.durans = None

    def slice(self, start, end):
        self.fnames = self.fnames[start:end]
        self.text_lengths = self.text_lengths[start:end]
        self.texts = self.texts[start:end]
        if self.embeds is not None:
            self.embeds = self.embeds[start:end]
        if self.durans is not None:
            self.durans = self.durans[start:end]

    def read_metadata(self, metadata_file):
        fnames, text_lengths, texts = [], [], []
        lines = codecs.open(metadata_file, 'r', 'utf-8').readlines()
        for line in lines:
            fname, _, text = line.strip().split("|")
            fnames.append(fname)
            text, tlen = self.text_proc(text)
            text_lengths.append(tlen[0])
            texts.append(np.array(text[0], np.long))
        return fnames, text_lengths, texts

    def load_durations(self, duration_file=None):
        duration_file = duration_file or osp.join(self.path, 'duration.txt')
        lines = codecs.open(duration_file, 'r', 'utf-8').readlines()
        self.durans = [[int(x) for x in l.split(',')] for l in lines]

    def load_embeddings(self, embedding_file=None):
        embedding_file = embedding_file or osp.join(self.path, 'embedding.txt')
        lines = codecs.open(embedding_file, 'r', 'utf-8').readlines()
        self.embeds = [[float(x) for x in l.split(',')] for l in lines]

    def get_test_data(self, texts, max_n=None):
        return self.text_proc(texts, max_n=max_n)

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        data = {}
        if 'texts' in self.keys:
            data['texts'] = self.texts[index]
        if 'tlens' in self.keys:
            data['tlens'] = np.array([data['texts'].shape[0]])
        if 'mels' in self.keys:
            # (T, 80)
            data['mels'] = np.load(os.path.join(self.path, 'mels', "%s.npy" % self.fnames[index]))
        if 'mels-gt' in self.keys:
            # (T, 80)
            data['mels'] = np.load(os.path.join(self.path, 'mels-gt', "%s.npy" % self.fnames[index]))
        if 'mlens' in self.keys:
            data['mlens'] = np.array([data['mels'].shape[0]])
        if 'embs' in self.keys:
            # (256,)
            data['embs'] = np.array(self.embeds[index])
        if 'drns' in self.keys:
            # (N,)
            data['drns'] = np.array(self.durans[index])
        return data
