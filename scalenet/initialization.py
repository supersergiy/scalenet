import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

###############
# Initializer #
###############
class _Initializer(nn.Module):
  def __init__(self, module, per_channel):
    super().__init__()
    self.module = module
    self.per_channel = per_channel
    self.initializing = False #True

  def initialize(self, x):
    raise NotImplementedError

  def forward(self, x):
    if self.initializing:
      self.initialize(x)
      self.initializing = False

    y = self.module(x)

    return y

  @property
  def weight(self):
    return self.module.weight

  @property
  def bias(self):
    return self.module.bias

  def expand_dims(self, x, ndim):
    for i in range(ndim-1):
      x = x.unsqueeze(-1)
    return x

  def mean(self, x, per_channel=False):
    c = x.shape[1]
    if per_channel:
      return x.transpose(0,1).contiguous().view(c, -1).mean(1).detach()
    else:
      return x.mean().detach()

  def std(self, x, per_channel=False):
    c = x.shape[1]
    if per_channel:
      return x.detach().transpose(0,1).contiguous().view(c, -1).var(1).mean().sqrt()
    else:
      return x.std().detach()

class Initializer(_Initializer):
  def __init__(self, module, per_channel=True, init_bias=True):
    super().__init__(module, per_channel)
    self.init_bias = init_bias

  def initialize(self, x):
    # init weight
    print ("INITIALIZE WEIGHT!!!!!!")
    nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
    y = self.module(x)
    sigma = self.std(y, per_channel=False)
    scale = 1.0 / sigma * np.sqrt(2.0)
    self.weight.data *= self.expand_dims(scale, self.weight.ndimension())

    # init bias
    if self.bias is not None:
      self.bias.data.zero_()
    if self.bias is not None and self.init_bias:
      y = self.module(x)
      mu = self.mean(y, self.per_channel)
      self.bias.data -= mu
