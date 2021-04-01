import torch
import numpy as np
import torch.nn.functional as F
from fastdtw import fastdtw
from .pytorch_ssim import ssim

from utils.functional import mask


class GuidedAttentionLoss(torch.nn.Module):
    """The guided attention loss from https://arxiv.org/abs/1710.08969"""
    def __init__(self, sigma=0.2):
        super(GuidedAttentionLoss, self).__init__()
        self.sigma = sigma  # strength of the loss

    def forward(self, attention_weights, len_rows, len_cols):
        """

        :param attention_weights: (batch, time_spectrograms, time_phonemes)
        :param len_rows:
        :param len_cols:
        :return:
        """

        B, T, N = attention_weights.shape  # batch, time, features
        y, x = torch.as_tensor(np.mgrid[0:T, 0:N]).float().to(attention_weights.device)

        t, n = torch.as_tensor(len_rows).float(), torch.as_tensor(len_cols).float()
        penalty = \
            1 - torch.exp(-(y[None, :, :] / t[:, None, None]
                            - x[None, :, :] / n[:, None, None]) ** 2 / (2 * self.sigma ** 2))

        mask = (y[None, :, :] < t[:, None, None]) * (x[None, :, :] < n[:, None, None])
        penalty = penalty.masked_fill(~mask, 0).to(attention_weights.device)

        l = penalty * attention_weights
        l = torch.sum(torch.sum(l, dim=-1), dim=-1) # sum each frame
        l /= torch.as_tensor(n * t).to(attention_weights.device) # divide each frame sum by total number of nonzero elements in the frame
        return torch.mean(l)


class L1Masked(torch.nn.Module):
    def __init__(self):
        super(L1Masked, self).__init__()
        self.l1 = torch.nn.L1Loss(reduction='sum')

    def forward(self, input, target, lengths):

        m = mask(input.shape, lengths, dim=1).float().to(input.device)
        return self.l1(input * m, target * m) / m.sum()


def l1_masked(input, target, lengths):
    m = mask(input.shape, lengths, dim=1).float().to(input.device)
    return F.l1_loss(input * m, target * m, reduction='sum') / m.sum()


def l1_dtw(input, ilen, target, tlen):
    """l1 loss after dynamic time warping of the corresponding spectrograms
    
    This is not a loss in true sence since it is not differentiable - it is a metric.
    :param input: (batch, time, channels)
    :param target: (batch, time, channels)
    :return:
    """

    input = [i[:l] for i, l in zip(input, ilen)]
    out = [t[:l] for t, l in zip(target, tlen)]

    s = 0
    total_elem = 0
    for i, t in zip(input, target):
        dtw, path = fastdtw(i.cpu(), t.cpu(), dist=lambda x, y: np.abs(x-y).sum())
        s += dtw
        if len(i) > len(t):
            total_elem += i.numel()
        else:
            total_elem += t.numel()

    return s / total_elem # divide by total number of elements


def guided_att(attention_weights, len_rows, len_cols, sigma=0.2):
        """

        :param attention_weights: (batch, time_spectrograms, time_phonemes)
        :param len_rows:
        :param len_cols:
        :return:
        """

        B, T, N = attention_weights.shape  # batch, time, features
        y, x = torch.as_tensor(np.mgrid[0:T, 0:N]).float().to(attention_weights.device)

        t, n = torch.as_tensor(len_rows).float(), torch.as_tensor(len_cols).float()
        penalty = \
            1 - torch.exp(-(y[None, :, :] / t[:, None, None]
                            - x[None, :, :] / n[:, None, None]) ** 2 / (2 * sigma ** 2))

        mask = (y[None, :, :] < t[:, None, None]) * (x[None, :, :] < n[:, None, None])
        penalty = penalty.masked_fill(~mask, 0).to(attention_weights.device)

        l = penalty * attention_weights
        l = torch.sum(torch.sum(l, dim=-1), dim=-1) # sum each frame
        l /= torch.as_tensor(n * t).to(attention_weights.device) # divide each frame sum by total number of nonzero elements in the frame
        return torch.mean(l)


def logit(x, eps=1e-8):
    return torch.log(x + eps) - torch.log(1 - x + eps)


def binary_divergence_masked(input, target, lengths):
    """ Provides non-vanishing gradient, but does not equal zero if spectrograms are the same
    Inspired by https://github.com/r9y9/deepvoice3_pytorch/blob/897f31e57eb6ec2f0cafa8dc62968e60f6a96407/train.py#L537
    """

    input_logits = logit(input)
    z = -target* input_logits + torch.log1p(torch.exp(input_logits))
    m = mask(input.shape, lengths, dim=1).float().to(input.device)

    return masked_mean(z, m)


def masked_mean(y, mask):
    # (batch, time channels)
    return (y * mask).sum() / mask.sum()


def masked_huber(input, target, lengths):
    """
    Always mask the first (non-batch dimension) -> usually time

    :param input:
    :param target:
    :param lengths:
    :return:
    """
    m = mask(input.shape, lengths, dim=1).float().to(input.device)
    return F.smooth_l1_loss(input * m, target * m, reduction='sum') / m.sum()


def masked_ssim(input, target, lengths):
    m = mask(input.shape, lengths, dim=1).float().to(input.device)
    input, target = input * m, target * m
    return 1-ssim(input.unsqueeze(1), target.unsqueeze(1))
