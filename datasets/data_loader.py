"""Data loader for speech dataset."""
import sys
import random

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate, DataLoader
from torch.utils.data.sampler import Sampler

__all__ = ['Text2MelDataLoader', 'SSRNDataLoader']


class Text2MelDataLoader(DataLoader):
    def __init__(self, text2mel_dataset, batch_size, mode='train', \
                 num_workers=0 if sys.platform.startswith('win') else 8, **kwargs):
        if mode == 'train':
            text2mel_dataset.slice(0, -batch_size)
        elif mode == 'valid':
            text2mel_dataset.slice(len(text2mel_dataset)-batch_size, -1)
        elif mode == 'whole':
            text2mel_dataset.slice(None, None)
        else:
            raise ValueError("mode must be either 'train' or 'valid' or 'whole'")
        super().__init__(text2mel_dataset,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         collate_fn=collate_fn,
                         shuffle=False, **kwargs)


class SSRNDataLoader(DataLoader):
    def __init__(self, ssrn_dataset, batch_size, mode='train', \
                 num_workers=0 if sys.platform.startswith('win') else 8):
        if mode == 'train':
            ssrn_dataset.slice(0, -batch_size)
            super().__init__(ssrn_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             collate_fn=collate_fn,
                             sampler=PartiallyRandomizedSimilarTimeLengthSampler(lengths=ssrn_dataset.text_lengths,
                                                                                 data_source=None,
                                                                                 batch_size=batch_size))
        elif mode == 'valid':
            ssrn_dataset.slice(len(ssrn_dataset) - batch_size, -1)
            super().__init__(ssrn_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             collate_fn=collate_fn,
                             shuffle=True)
        else:
            raise ValueError("mode must be either 'train' or 'valid'")


def collate_fn(batch):
    keys = batch[0].keys()
    max_lengths = {key: 0 for key in keys}
    collated_batch = {key: [] for key in keys}

    # find out the max lengths
    for row in batch:
        for key in keys:
            max_lengths[key] = max(max_lengths[key], row[key].shape[0])

    # pad to the max lengths
    for row in batch:
        for key in keys:
            array = row[key]
            dim = len(array.shape)
            assert dim == 1 or dim == 2
            # TODO: because of pre processing, later we want to have (n_mels, T)
            if dim == 1:
                padded_array = np.pad(array, (0, max_lengths[key] - array.shape[0]), mode='constant')
            else:
                padded_array = np.pad(array, ((0, max_lengths[key] - array.shape[0]), (0, 0)), mode='constant')
            collated_batch[key].append(padded_array)

    # use the default_collate to convert to tensors
    for key in keys:
        collated_batch[key] = default_collate(collated_batch[key])
    return collated_batch


class PartiallyRandomizedSimilarTimeLengthSampler(Sampler):
    """Copied from: https://github.com/r9y9/deepvoice3_pytorch/blob/master/train.py.
    Partially randomized sampler
    1. Sort by lengths
    2. Pick a small patch and randomize it
    3. Permutate mini-batches
    """

    def __init__(self, lengths, data_source, batch_size=16, batch_group_size=None, permutate=True):
        super().__init__(data_source)
        self.lengths, self.sorted_indices = torch.sort(torch.LongTensor(lengths))
        self.batch_size = batch_size
        if batch_group_size is None:
            batch_group_size = min(batch_size * 32, len(self.lengths))
            if batch_group_size % batch_size != 0:
                batch_group_size -= batch_group_size % batch_size

        self.batch_group_size = batch_group_size
        assert batch_group_size % batch_size == 0
        self.permutate = permutate

    def __iter__(self):
        indices = self.sorted_indices.clone()
        batch_group_size = self.batch_group_size
        s, e = 0, 0
        for i in range(len(indices) // batch_group_size):
            s = i * batch_group_size
            e = s + batch_group_size
            random.shuffle(indices[s:e])

        # Permutate batches
        if self.permutate:
            perm = np.arange(len(indices[:e]) // self.batch_size)
            random.shuffle(perm)
            indices[:e] = indices[:e].view(-1, self.batch_size)[perm, :].view(-1)

        # Handle last elements
        s += batch_group_size
        if s < len(indices):
            random.shuffle(indices[s:])

        return iter(indices)

    def __len__(self):
        return len(self.sorted_indices)
