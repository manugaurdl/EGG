import sys
import os
import shutil
import warnings
import requests
import pidfile
from contextlib import contextmanager
from time import sleep

import torch
from torch import Tensor

@contextmanager
def exclusive(pidname):
    done = False
    while not done:
        try:
            with pidfile.PIDFile(pidname):
                yield
                done = True
        except pidfile.AlreadyRunningError:
            sleep(5)


def download_from_url(url, path):
    """Download file, with logic (from tensor2tensor) for Google Drive"""
    if 'drive.google.com' not in url:
        print('Downloading %s; may take a few minutes' % url)
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        with open(path, "wb") as file:
            file.write(r.content)
        return
    print('Downloading from Google Drive; may take a few minutes')
    confirm_token = None
    session = requests.Session()
    response = session.get(url, stream=True)
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v

    if confirm_token:
        url = url + "&confirm=" + confirm_token
        response = session.get(url, stream=True)

    chunk_size = 16 * 1024
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)


def one_hot_to_index(one_hot: Tensor) -> Tensor:
    """
    Converts a one-hot tensor into a tensor with corresponding indexes
    """
    device, dtype = one_hot.device, one_hot.dtype
    vocab_size = one_hot.shape[-1]
    oh2idx = torch.tensor(range(vocab_size), dtype=dtype, device=device)
    return (one_hot @ oh2idx.unsqueeze(dim=1)).long().squeeze(dim=-1)

class DummyFile(object):
    def write(self, x): pass


@contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout
