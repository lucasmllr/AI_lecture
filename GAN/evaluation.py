import training
import pylab as plt
import numpy as np
import os
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils import data
from model import Generator

def imshow(inp, path=None, title=None, show=False):
    """Imshow for Tensor."""

    fig = plt.figure(figsize=(5, 5))
    inp = inp.numpy()
    inp = inp.transpose((1, 2, 0))
    plt.imshow(inp)

    if title:
        plt.title(title)

    if path and not os.path.exists('path'):
        plt.savefig(path, bbox_inches='tight')
        plt.close(fig)

    if show:
        plt.show()

    return
