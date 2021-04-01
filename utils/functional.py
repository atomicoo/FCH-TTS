import numpy as np
import torch


def mask(shape, lengths, dim=-1):

    assert dim != 0, 'Masking not available for batch dimension'
    assert len(lengths) == shape[0], 'Lengths must contain as many elements as there are items in the batch'

    lengths = torch.as_tensor(lengths)

    to_expand = [1] * (len(shape)-1)+[-1]
    mask = torch.arange(shape[dim]).expand(to_expand).transpose(dim, -1).expand(shape).to(lengths.device)
    mask = mask < lengths.expand(to_expand).transpose(0, -1)
    return mask


def positional_encoding(channels, length, w=1):
    """The positional encoding from `Attention is all you need` paper

    :param channels: How many channels to use
    :param length: 
    :param w: Scaling factor
    :return:
    """
    enc = torch.FloatTensor(length, channels)
    rows = torch.arange(length, out=torch.FloatTensor())[:, None]
    cols = 2 * torch.arange(channels//2, out=torch.FloatTensor())

    enc[:, 0::2] = torch.sin(w * rows / (10.0**4 ** (cols / channels)))
    enc[:, 1::2] = torch.cos(w * rows / (10.0**4 ** (cols / channels)))
    return enc


def scaled_dot_attention(q, k, v, mask=None, noise=0, dropout=lambda x: x):
    """
    :param q: queries, (batch, time1, channels1)
    :param k: keys, (batch, time2, channels1)
    :param v: values, (batch, time2, channels2)
    :param mask: boolean mask, (batch, time1, time2)
    :param dropout: a dropout function - this allows keeping dropout as a module -> better control when training/eval
    :return: (batch, time1, channels2), (batch, time1, time2)
    """

    # (batch, time1, time2)
    weights = torch.matmul(q, k.transpose(2, 1))
    if mask is not None:
        weights = weights.masked_fill(~mask, float('-inf'))

    if noise:
        weights += noise * torch.randn(weights.shape).to(weights.device)

    weights = torch.softmax(weights, dim=-1)
    weights = dropout(weights)

    result = torch.matmul(weights, v)  # (batch, time1, channels2)
    return result, weights


def get_alphabet_durations(alignment):
    """Alignment must be a batch

    :return counts: [(idx, count), ...]
    """

    alignment = torch.as_tensor(alignment)
    maxx = torch.max(alignment, dim=-1)[1]
    counts = [torch.unique(m, return_counts=True) for m in maxx]

    return counts


def pad_batch(items, pad_value=0):
    """Pad tensors in list to equal length (in the first dim)

    :param items:
    :param pad_value:
    :return: padded_items, orig_lens
    """
    max_len = len(max(items, key=lambda x: len(x)))
    zeros = (2*torch.as_tensor(items[0]).ndim -1) * [pad_value]
    padded_items = torch.stack([torch.nn.functional.pad(torch.as_tensor(x), pad= zeros + [max_len - len(x)], value=pad_value)
                          for x in items])
    orig_lens = [len(xx) for xx in items]
    return padded_items, orig_lens


def get_fertilities(alignments, plen, slen):
    """Smoothed fertilities

    Values at indices correspond to fertilities for the phoneme at the given index.

    :param alignments: (batch, time, phoneme_len)
    :param plen: original phoneme length of each sentence in batch before padding
    :param slen: original spectrogram length before padding
    :return: list of 1D numpy arrays
    """
    fert = fertilities_improper(alignments, plen, slen)
    smoothed = smooth_fertilities(fert, slen)
    return smoothed


def fertilities_improper(alignments, plen, slen):
    """Phonemes not attended to get fertility one -> sum of fertilities may not equal slen

    Apply smoothing to get fertilities where sum of fertilities corresponds to number of spetrogram frames
    alignments must be non-decreasing! Enforce eg by monotonic attention

    :param alignments: (batch, time, phoneme_len)
    :return: fertilities: list of tensors
    """

    fertilities = []
    for i, a in enumerate(alignments):
        a = a[:slen[i], :plen[i]]
        # if frame is full of zeros, the attention went outside allowed range.
        # Att is monotonic -> place 1 to the end because the att will never come back -> focus on the last phoneme
        a[~(a>0).any(dim=1), -1] = 1
        am, _ = torch.argmax(a, dim=-1).sort()
        # expects sorted array
        uniq, counts = torch.unique_consecutive(am, return_counts=True)
        fert = torch.ones(plen[i], dtype=torch.long).to(alignments.device)  # bins for each phoneme
        fert[uniq] = counts
        fertilities.append(fert)

    return fertilities


def smooth_fertilities(fertilities_improper, slen):
    """Uniformly subtract 1 from n largest fertility bins, where n is the number of extra fertility points

    After smoothing, we should have sum(fertilities) = slen

    :param raw_fertilities: List of tensors from `fertilities_raw`
    :param slen: spectrogram lens
    :return: smooth_fertilities
    """

    smoothed = []
    for i, f in enumerate(fertilities_improper):
        f = f.detach().cpu().numpy()
        ff = f.copy()
        frames = slen[i]
        extra = ff.sum() - frames
        if extra:
            n_largest = np.argpartition(f, -extra)[-extra:]  # get `extra` largest fertilities indices
            ff[n_largest] -= 1
        smoothed.append(ff)

    return smoothed


def load_alignments(file):
    with open(file) as f:
        alignments = [[int(x) for x in l.split(',')] for l in f.readlines()]
    return alignments


def fert2align(fertilities):
    """Map list of fertilities to alignment matrix

    Allows backwards mapping for sanity check.

    :param fertilities: list of lists
    :return: alignment, list of numpy arrays, shape (batch, slen, plen)
    """

    alignments = []
    for f in fertilities:
        frames = np.sum(f.astype(int))
        a = np.zeros((frames, len(f)))
        x = np.arange(frames)
        y = np.repeat(np.arange(len(f)), f.astype(int))  # repeat each phoneme index according to fertiities
        a[(x, y)] = 1
        alignments.append(a)

    return alignments


# def pad_list(xs, pad_value):
#     """Perform padding for the list of tensors.
#     Args:
#         xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
#         pad_value (float): Value for padding.
#     Returns:
#         Tensor: Padded tensor (B, Tmax, `*`).
#     Examples:
#         >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
#         >>> x
#         [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
#         >>> pad_list(x, 0)
#         tensor([[1., 1., 1., 1.],
#                 [1., 1., 0., 0.],
#                 [1., 0., 0., 0.]])
#     """
#     n_batch = len(xs)
#     max_len = max(x.size(0) for x in xs)
#     pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

#     for i in range(n_batch):
#         pad[i, : xs[i].size(0)] = xs[i]

#     return pad

# def repeat_one_sequence(x, d):
#     """Repeat each frame according to duration for torch 1.1+."""
#     return torch.repeat_interleave(x, d, dim=0)

# def repeat_batch_sequence(xs, ds, pad_value=0):
#     """Repeat each frame according to duration for torch 1.1+."""
#     return pad_list([repeat_one_sequence(x, d) for x, d in zip(xs, ds)], pad_value)

