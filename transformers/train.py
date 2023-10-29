import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
from itertools import zip_longest

def get_dict(filename, pad=0, bos=1, eos=2, unk=3):
    token_map = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<EOS>": 3}
    with open(filename) as f:
        for i, l in enumerate(f, start=4):
            keys = l.strip().split()
            token_map[keys[0]] = i
    return token_map

def batch_padding(batch, padding_idx=0):
    max_len = len(max(batch, key=lambda x: len(x)))
    for sent in batch:
        padding_len = max_len - len(sent)
        if padding_len:
            sent.extend([padding_idx] * padding_len)
    return batch

def real_data_gen(V, batch, nbatches):
    dict_z0.
    
