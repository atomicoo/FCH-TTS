import torch.nn as nn
import torch.nn.functional as F
from torch import as_tensor, stack, clamp
from ignite.utils import apply_to_tensor


# TODO: add inverse transforms
class Transform:
    """Compose transforms

    Example:

    t1 = [1.,2.,3.,4.]
    t2 = [1.,2.,3.,4.,5.,6.,7.]
    t = [t1, t2]

    pipe = Transform([
        map_to_tensors,
        Clamp(-85, 0),
        Normalize(mode='min-max', min=1, max=7, a=0, b=1),
        Pad(0)
    ])

    x = pipe(t)
    print(x)
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class Pad:
    """Pad all tensors in first (length) dimension"""

    def __init__(self, pad_value=0, get_lens=False):
        self.pad_value = pad_value
        self.get_lens = get_lens

    def __call__(self, x):
        """Pad each tensor in x to the same length

        Pad tensors in the first dimension and stack them to form a batch

        :param x: list of tensors/lists/arrays
        :returns batch: (len_x, max_len_x, ...)
        """

        if self.get_lens:
            return self.pad_batch(x, self.pad_value), [len(xx) for xx in x]

        return self.pad_batch(x, self.pad_value)

    @staticmethod
    def pad_batch(items, pad_value=0):
        max_len = len(max(items, key=lambda x: len(x)))
        zeros = (2*as_tensor(items[0]).ndim -1) * [pad_value]
        return stack([F.pad(as_tensor(x), pad= zeros + [max_len - len(x)], value=pad_value)
                      for x in items])


class Normalize:

    MODES = ['min-max', 'standardize']

    def __init__(self, mode="min-max",
                 min=None, max=None, a=0, b=1,
                 mean=None, std=None):

        assert mode in self.MODES, "Invalid mode"
        self.mode = mode

        if self.mode == 'standardize':
            assert mean is not None and std is not None
            self.mean = mean
            self.std = std

        if self.mode == 'min-max':
            assert min is not None and max is not None
            self.min, self.max = min, max
            self.a, self.b = a, b

    def __call__(self, x, inverse=False):

        if inverse:
            if self.mode == 'min-max': return apply_to_tensor(x, self.norm_min_max_inv)
            if self.mode == 'standardize': return apply_to_tensor(x, self.norm_standard_inv)
        else:
            if self.mode == 'min-max': return apply_to_tensor(x, self.norm_min_max)
            if self.mode == 'standardize': return apply_to_tensor(x, self.norm_standard)

    def norm_min_max(self, x):
        x = as_tensor(x)
        return self.a + (x - self.min) * (self.b - self.a) / (self.max - self.min)

    def norm_min_max_inv(self, x):
        x = as_tensor(x)
        return self.min + (x - self.a) * (self.max - self.min) / (self.b - self.a)

    def norm_standard(self, x):
        return (x - self.mean)/self.std

    def norm_standard_inv(self, x):
        x = as_tensor(x)
        return x * self.std + self.mean


def map_to_tensors(x_list):
    return [as_tensor(x) for x in x_list]


class MinMaxNorm(nn.Module):
    def __init__(self, min, max, a=0, b=1):
        super(MinMaxNorm, self).__init__()
        self.min, self.max = min, max
        self.a, self.b = a, b

    def forward(self, x):
        return self.a + (x - self.min) * (self.b - self.a) / (self.max - self.min)

    def inverse(self, x):
        return self.min + (x - self.a) * (self.max - self.min) / (self.b - self.a)


class StandardNorm(nn.Module):
    def __init__(self, mean, std):
        super(StandardNorm, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return (x - self.mean)/self.std

    def inverse(self, x):
        return x * self.std + self.mean



class Clamp(nn.Module):

    def __init__(self, floor, ceil):
        self.floor, self.ceil = floor, ceil

    def __call__(self, x):
        return apply_to_tensor(x, lambda y: clamp(as_tensor(y), self.floor, self.ceil))
