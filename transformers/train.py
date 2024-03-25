import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from typing import Any
import matplotlib.pyplot as plt
from itertools import zip_longest
import transformers_define as ts

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
            ts.subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
    
    def rate(self, step = None):
        if step is None:
            step = self._step
        
        return self.factor * (self.model_size ** (-0.5) * min(
            step ** (-0.5), step * self.warmup ** (-1.5)))
    
class SimpleLossCompute:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item() * norm


def get_dict(filename, pad=0, bos=1, eos=2, unk=3):
    token_map = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<EOS>": 3}
    with open(filename, encoding='utf-8') as f:
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
    dict_zh = get_dict(r"D:\PY学习\transformers\transformers\data/dict.zh-cn")
    dict_en = get_dict(r"D:\PY学习\transformers\transformers\data/dict.en")

    train_en = open(r"D:\PY学习\transformers\transformers\data/train.en.bped", encoding='utf-8')
    train_zh = open(r"D:\PY学习\transformers\transformers\data/train.zh-cn.bped", encoding='utf-8')

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


def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = ts.MultiHeadedAttention(h, d_model)
    ff = ts.PositonwiseFeedForward(d_model, d_ff, dropout)
    position = ts.PositionalEncoding(d_model, dropout)
    model = ts.EncoderDecoder(
        ts.Encoder(ts.EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        ts.Decoder(ts.DecoderLayer(d_model, c(attn), c(attn), c(ff),dropout), N),
        nn.Sequential(ts.Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(ts.Embeddings(d_model, tgt_vocab), c(position)),
        ts.Generator(d_model, tgt_vocab))
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

def run_epoch(data_iter, model, loss_compute):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec : %f" % 
                  (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens

V_zh = 12203 + 2
V_en = 30684 + 2
V = V_en
criterion = LabelSmoothing(size=V_zh, padding_idx=0, smoothing=0.0)
model = make_model(V_en, V_zh, N=2)
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

for epoch in range(10):
    model.train()
    run_epoch(real_data_gen(V, 30, 20), model, SimpleLossCompute(model.generator, criterion, model_opt))
    model.eval()
    print(run_epoch(real_data_gen(V, 30, 5), model, SimpleLossCompute(model.generator, criterion, None)))
    



    
