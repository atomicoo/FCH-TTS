import torch
from random import sample, randrange


def add_random_noise(specs, std_dev):
    """Add noise from Normal(0, std_dev)

    :param specs:
    :param std_dev:
    :return:
    """
    if not std_dev: return specs
    return specs + std_dev * torch.randn(specs.shape).to(specs.device)


def degrade_some(model, specs, texts, tlens, ratio, repeat=1):
    """Replace some spectrograms in batch by their generated equivalent

    Ideally, run this after adding random noise
    so that the generated spectrograms are slightly degenerated.

    :param ratio: How many percent of spectrograms in batch to degrade (0,1)
    :param repeat: How many times to degrade
    :return:
    """
    if not ratio: return specs
    if not repeat: return specs

    idx = sample(range(len(specs)), int(ratio * len(specs)))

    with torch.no_grad():
        s = specs
        for i in range(repeat):
            s, *_ = model((texts, tlens, specs, True))

        specs[idx] = s[idx]

    return specs


def replace_frames_with_random(specs, ratio, distrib=torch.rand):
    """

    Each spectrogram gets different frames degraded.
    To use normal noise, set distrib=lambda shape: mean + std_dev * torch.randn(x)

    :param specs:
    :param ratio: between 0,1 - how many percent of frames to degrade
    :param distrib: default torch.rand -> [0, 1 uniform]
    :return:
    """
    if not ratio: return specs

    t = specs.shape[1]
    num_frames = int(t * ratio)
    idx = [sample(range(t), num_frames) for i in range(len(specs))]  # different for each spec.

    for s, _ in enumerate(specs):
        rnd_frames = distrib((num_frames, specs.shape[-1])).to(specs.device)
        specs[s, idx[s]] = rnd_frames

    return specs


def frame_dropout(specs, ratio):
    """Replace random frames with zeros

    :param specs:
    :param ratio:
    :return:
    """
    return replace_frames_with_random(specs, ratio, distrib=lambda shape: torch.zeros(shape))


def random_patches(specs1, specs2, width, slen):
    """Create random patches from spectrograms

    :param specs: (batch, time, channels)
    :param width: int
    :param slen: list of int
    :return: patches (batch, width, channels)
    """

    idx = [randrange(l - width) for l in slen]
    patches1, patches2 = [s[i:i+width] for s, i in zip(specs1, idx)], [s[i:i+width] for s, i in zip(specs2, idx)]
    return torch.stack(patches1), torch.stack(patches2)
