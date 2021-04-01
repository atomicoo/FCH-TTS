"""Utility methods."""
__author__ = 'Erdene-Ochir Tuguldur'

import os
import sys
import glob
import torch
import math
import requests
from tqdm import tqdm
import numpy as np
from skimage.io import imsave
from skimage import img_as_ubyte


def select_device(device):
    cpu_request = device.lower() == 'cpu'
    # if device requested other than 'cpu'
    if device and not cpu_request:
        c = 1024 ** 2  # bytes to MB
        x = torch.cuda.get_device_properties(int(device))
        s = f'Using torch {torch.__version__} '
        print("%sCUDA:%s (%s, %dMB)" % (s, device, x.name, x.total_memory / c))
        return torch.device(f'cuda:{device}')
    else:
        print(f'Using torch {torch.__version__} CPU')
        return torch.device('cpu')


def get_last_chkpt_path(logdir):
    """Returns the last checkpoint file name in the given log dir path."""
    checkpoints = glob.glob(os.path.join(logdir, '*.pth'))
    checkpoints.sort()
    if len(checkpoints) == 0:
        return None
    return checkpoints[-1]


def load_checkpoint(checkpoint_file_name, model, optimizer, device):
    """Loads the checkpoint into the given model and optimizer."""
    checkpoint = torch.load(checkpoint_file_name, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.float()
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('step', 0)
    del checkpoint
    print("Loaded checkpoint epoch=%d step=%d" % (epoch, step))
    return epoch, step


def save_checkpoint(logdir, epoch, global_step, model, optimizer):
    """Saves the training state into the given log dir path."""
    checkpoint_file_name = os.path.join(logdir, 'step-%03dK.pth' % (global_step // 1000))
    print("Saving the checkpoint file '%s'..." % checkpoint_file_name)
    checkpoint = {
        'epoch': epoch + 1,
        'step': global_step,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_file_name)
    del checkpoint


def download_file(url, file_path):
    """Downloads a file from the given URL."""
    print("Downloading %s..." % url)
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024 * 1024
    wrote = 0
    with open(file_path, 'wb') as f:
        for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size // block_size), unit='MB'):
            wrote = wrote + len(data)
            f.write(data)

    if total_size != 0 and wrote != total_size:
        print("Downloading failed")
        sys.exit(1)


def save_to_png(file_name, array):
    """Save the given numpy array as a PNG file."""
    # from skimage._shared._warnings import expected_warnings
    # with expected_warnings(['precision']):
    imsave(file_name, img_as_ubyte(array))
