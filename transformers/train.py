import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
from itertools import zip_longest
from bin.transformers import subsequent_mask

class Batch:
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1).type_as(tgt_mask.data))
        )
        return tgt_mask

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
    dict_zh = get_dict("dict.zh-cn")
    dict_en = get_dict("dict.en")

    train_en = open("./data/train.en.bped")
    train_zh = open("./data/train.zh-cn.bped")

    batch_en = []
    batch_zh = []

    for sent_en, sent_zh in zip(train_en, train_zh):
        sent_en = "<BOS> {} <EOS>".format(sent_en.strip())
        sent_zh = "<BOS> {} <EOS>".format(sent_zh.strip())
        batch_en.append([dict_en[token] for token in sent_en.split()])
        batch_zh.append([dict_zh[token] for token in sent_zh.split()])

        if len(batch_en) % batch == 0:
            src = torch.tensor(batch_padding(batch_en, 0), dtype=torch.int)
            tgt = torch.tensor(batch_padding(batch_zh, 0), dtype=torch.int)
            src = src.long()
            tgt = tgt.long()
            yield Batch(src, tgt, 0)
        else:
            src = src.long()
            tgt = tgt.long()
            yield Batch(src, tgt, 0)

V_zh = 12203 + 2
V_en = 30684 + 2
V = V_en


    
