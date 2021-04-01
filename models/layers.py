import torch.nn as nn
import torch
from torch import tanh, sigmoid

from utils.functional import scaled_dot_attention
from torch.nn import LayerNorm


class Conv1d(nn.Conv1d):
    """A wrapper around nn.Conv1d, that works on (batch, time, channels)"""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, groups=1, bias=True, padding=0):
        super(Conv1d, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=kernel_size, stride=stride, dilation=dilation,
                                     groups=groups, bias=bias, padding=padding)

    def forward(self, x):
        return super().forward(x.transpose(2,1)).transpose(2,1)


class FreqNorm(nn.BatchNorm1d):
    """Normalize separately each frequency channel in spectrogram and batch,


    Examples:
        t = torch.arange(2*10*5).reshape(2, 10, 5).float()
        b1 = nn.BatchNorm1d(10, affine=False, momentum=None)
        b2 = (t - t.mean([0,2], keepdim=True))/torch.sqrt(t.var([0,2], unbiased=False, keepdim=True)+1e-05)
        -> b1 and b2 give the same results
        -> BatchNorm1D by default normalizes over channels and batch - not useful for differet length sequences
        If we transpose last two dims, we get normalizaton across batch and time
        -> normalization for each frequency channel over time and batch

        # compare to layer norm:
        Layer_norm: (t - t.mean(-1, keepdim=True))/torch.sqrt(t.var(-1, unbiased=False, keepdim=True)+1e-05)
        -> layer norm normalizes across all frequencies for each timestep independently of batch

        => LayerNorm: Normalize each freq. bin wrt to other freq bins in the same timestep -> time independent, batch independent, freq deendent
        => FreqNorm: Normalize each freq. bin wrt to the same freq bin across time and batch -> time dependent, other freq independent
    """
    def __init__(self, channels, affine=True, track_running_stats=True, momentum=0.1):
        super(FreqNorm, self).__init__(channels, affine=affine, track_running_stats=track_running_stats, momentum=momentum)

    def forward(self, x):
        return super().forward(x.transpose(2,1)).transpose(2,1)


class ResidualBlock(nn.Module):
    """Implements conv->PReLU->norm n-times"""

    def __init__(self, channels, kernel_size, dilation,  n=2, norm=FreqNorm, activation=nn.ReLU):
        super(ResidualBlock, self).__init__()

        self.blocks = [
            nn.Sequential(
                Conv1d(channels, channels, kernel_size, dilation=dilation),
                ZeroTemporalPad(kernel_size, dilation),
                activation(),
                norm(channels),  # Normalize after activation. if we used ReLU, half of our neurons would be dead!
            )
            for i in range(n)
        ]

        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        return x + self.blocks(x)


class ScaledDotAttention(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, noise=0, normalize=False, dropout=False):
        super(ScaledDotAttention, self).__init__()

        self.noise = noise
        self.dropout = torch.nn.Dropout(p=dropout)

        self.normalize = normalize
        self.fc_query = Conv1d(in_channels, hidden_channels)
        self.fc_keys = Conv1d(in_channels, hidden_channels)

        if normalize:
            self.qnorm = LayerNorm(in_channels)
            self.knorm = LayerNorm(in_channels)

        self.fc_keys.weight = torch.nn.Parameter(self.fc_query.weight.clone())
        self.fc_keys.bias = torch.nn.Parameter(self.fc_query.bias.clone())

        self.fc_values = Conv1d(in_channels, hidden_channels)
        self.fc_out = Conv1d(hidden_channels, out_channels)

    def forward(self, q, k, v, mask=None):
        """
        :param q: queries, (batch, time1, channels1)
        :param k: keys, (batch, time2, channels1)
        :param v: values, (batch, time2, channels2)
        :param mask: boolean mask, (batch, time1, time2)
        :return: (batch, time1, channels2), (batch, time1, time2)
        """

        noise = self.noise if self.training else 0

        if self.normalize:
            q = self.qnorm(q)
            k = self.knorm(k)

        alignment, weights = scaled_dot_attention(self.fc_query(q),
                                                  self.fc_keys(k),
                                                  self.fc_values(v),
                                                  mask, noise=noise, dropout=self.dropout)
        alignment = self.fc_out(alignment)
        return alignment, weights


class Pad(nn.ZeroPad2d):
    def __init__(self, kernel_size, dilation):
        pad_total = dilation * (kernel_size - 1)
        begin = pad_total // 2
        end = pad_total - begin

        super(Pad, self).__init__((begin, end, begin, end))


class ZeroTemporalPad(nn.ZeroPad2d):
    """Pad sequences to equal lentgh in the temporal dimension"""
    def __init__(self, kernel_size, dilation, causal=False):
        total_pad = (dilation * (kernel_size - 1))

        if causal:
            super(ZeroTemporalPad, self).__init__((0, 0, total_pad, 0))
        else:
            begin = total_pad // 2
            end = total_pad - begin
            super(ZeroTemporalPad, self).__init__((0, 0, begin, end))


class WaveResidualBlock(nn.Module):
    """A residual gated block based on WaveNet
                        |-------------------------------------------------------------|
                        |                                                             |
                        |                        |-- conv -- tanh --|                 |
          residual ->  -|--(pos_enc)--(dropout)--|                  * ---|--- 1x1 --- + --> residual
                                                 |-- conv -- sigm --|    |
                                                                        1x1
                                                                         |
          -------------------------------------------------------------> + ------------> skip
    """
    def __init__(self, residual_channels, block_channels, kernel_size, dilation_rate, causal=True, dropout=False, skip_channels=False):
        """
        :param residual_channels: Num. of channels for resid. connections between wave blocks
        :param block_channels: Num. of channels used inside wave blocks
        :param kernel_size: Num. of branches for each convolution kernel
        :param dilation_rate: Hom much to dilate inputs before applying gate and filter
        :param causal: If causal, input is zero padded from the front, else both sides are zero padded equally
        :param dropout: If dropout>0, apply dropout on the input to the block gates (not the residual connection)
        :param skip_channels: If >0, return also skip (batch, time, skip_channels)
        """
        super(WaveResidualBlock, self).__init__()

        self.pad = ZeroTemporalPad(kernel_size, dilation_rate, causal=causal)
        self.causal = causal
        self.receptive_field = dilation_rate * (kernel_size - 1) + 1

        # tanh and sigmoid applied in forward
        self.dropout = torch.nn.Dropout(p=dropout) if dropout else None
        self.filter = Conv1d(residual_channels, block_channels, kernel_size, dilation=dilation_rate)
        self.gate = Conv1d(residual_channels, block_channels, kernel_size, dilation=dilation_rate)

        self.conv1x1_resid = Conv1d(block_channels, residual_channels, 1)
        self.conv1x1_skip = Conv1d(block_channels, skip_channels, 1) if skip_channels else None

        self.tensor_q = None
        self.generate = False

    def forward(self, residual):
        """Feed residual through the WaveBlock
        
        Allows layer-level caching for faster sequential inference.
        See https://github.com/tomlepaine/fast-wavenet for similar tensorflow implementation and original paper.

        Non - causal version does not support iterative generation for obvious reasons.
        WARNING: generating must be called before each generated sequence!
        Otherwise there will be an error due to stored queue from previous run.

        RuntimeError: Trying to backward through the graph a second time,
        but the buffers have already been freed. Specify retain_graph=True when calling backward the first time.

        :param residual: Residual from previous block or from input_conv, (batch_size, channels, time_dim)
        :return: residual, skip
        """

        if self.generate and self.causal:
            if self.tensor_q is None:
                x = self.pad(residual)
                self.tensor_q = x[:, -self.receptive_field:, :].detach()
            else:
                assert residual.shape[1] == 1, f'Expected residual.shape[1] == 1 during generation, but got residual.shape[1]={residual.shape[1]}'

                x = torch.cat((self.tensor_q, residual), dim=1)[:, -self.receptive_field:, :]
                self.tensor_q = x.detach()
        else:
            x = self.pad(residual)

        if self.dropout is not None:
            x = self.dropout(x)
        filter = tanh(self.filter(x))
        gate = sigmoid(self.gate(x))
        out = filter * gate
        residual = self.conv1x1_resid(out) + residual

        if self.conv1x1_skip is not None:
            return residual, self.conv1x1_skip(out)
        else:
            return residual

    def generating(self, mode):
        """Call before and after generating"""
        self.generate = mode
        self.reset_queue()

    def reset_queue(self):
        self.tensor_q = None
