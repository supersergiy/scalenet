from torch import nn
from torch.autograd import Variable
import numpy as np
import torch
import six
import os
import random

from .model import Model
from .tools import dict_to_str
from .initialization import Initializer

from pdb import set_trace as st

class TissueNorm():
    def __init__(self):
        self.normer = torch.nn.InstanceNorm1d(1)
    def __call__(self, x):
        mask = x != 0
        result = x.clone()
        result[mask] = self.normer(x[mask].unsqueeze(0)).squeeze(0)

class Sequence(Model):
    def initc(self, m, mult):
        #nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.bias.data[...] = 0
        if False:
            m.weight.data *= mult

        return m

    def validate_arch_desc(self, arch_desc):
        #TODO
        return

    def __init__(self, arch_desc, batchnorm=False):
        super().__init__()
        self.batchnorm = batchnorm
        self.name = dict_to_str(arch_desc)
        self.params = self.parse_arch_desc(arch_desc)
        self.construct_layers(self.params)
        #self.do_dynamic_init = torch.nn.Parameter(torch.zeros(2, requires_grad=False))
        self.init_bias = True
        self.per_channel = True

    def parse_arch_desc(self, arch_desc):
        self.validate_arch_desc(arch_desc)
        params = {
            'flags': {
                "use_batchnorm": self.batchnorm,
                "use_inputnorm": False,
                "use_tissuenorm_in": False,
                "use_tissuenorm_all": False,
                "use_instancenorm": False
            },
            'conv': {
                'k': 7,
                'init_mult': np.sqrt(6)
            },
            'act': {
                'constructor': nn.LeakyReLU
            },
            'structure': {
                'fms': None,
                'skips': {}
            }
        }

        self.parse_conv_params(params, arch_desc)
        self.parse_act_params(params, arch_desc)
        self.parse_structure_params(params, arch_desc)
        self.parse_flag_params(params, arch_desc)
        return params

    def construct_layers(self, params):
        fms = params['structure']['fms']
        skips = params['structure']['skips']

        init_mult = params['conv']['init_mult']
        k = params['conv']['k']
        pad = (k - 1) // 2

        act_constructor = params['act']['constructor']
        self.layers = torch.nn.ModuleList()

        if params['flags']['use_inputnorm']:
            self.layers.append(torch.nn.InstanceNorm2d(num_features=fms[0]))

        if params['flags']['use_inputnorm']:
            self.layers.append(TissueNorm(num_features=fms[0]))

        for i in range(len(fms) - 1):
            self.layers.append(nn.Conv2d(fms[i], fms[i + 1], k, padding=pad))
            self.initc(self.layers[-1], init_mult)

            if i != len(fms) - 2:
                if params['flags']['use_batchnorm']:
                    self.layers.append(nn.BatchNorm2d(fms[i + 1]))
                if params['flags']['use_instancenorm']:
                    self.layers.append(torch.nn.InstanceNorm2d(
                                                  num_features=fms[i + 1]))
                self.layers.append(act_constructor())
        self.seq = nn.Sequential(*self.layers)

    def parse_conv_params(self, params, arch_desc):
        if 'k' in arch_desc:
            params['conv']['k'] = arch_desc['k']

        if 'conv_init_mult' in arch_desc:
            params['conv']['init_mult'] = arch_desc['conv_init_mult']

        if 'initc_mult' in arch_desc:
            params['conv']['init_mult'] = arch_desc['initc_mult']

    def parse_act_params(self, params, arch_desc):
        if 'act' in arch_desc:
            act = arch_desc['act']
            if act == 'lrelu':
                params['act']['constructor'] = nn.LeakyReLU
            elif act == 'tanh':
                params['act']['constructor'] = nn.Tanh
            if act == 'relu':
                params['act']['constructor'] = nn.ReLU
            else:
                raise Exception("Unrecognized Activation")

    def parse_flag_params(self, params, arch_desc):
        if 'flags' in arch_desc and 'batchnorm' in arch_desc['flags']:
            params['flags']['use_batchnorm'] = True

        if 'flags' in arch_desc and 'instancenorm' in arch_desc['flags']:
            params['flags']['use_instancenorm'] = True

        if 'flags' in arch_desc and 'inputnorm' in arch_desc['flags']:
            params['flags']['use_inputnorm'] = True

    def parse_structure_params(self, params, arch_desc):
        params['structure']['fms'] = arch_desc['fms']
        if 'skips' in arch_desc:
            params['structure']['skips'] = {
                    int(s): e for (s, e) in six.iteritems(arch_desc['skips'])
            }

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


    def dynamic_initialize(self, l, x):
        # init weight
        print ("INITIALIZE WEIGHT!!!!!!")
        y = l(x)
        sigma = self.std(y, per_channel=False)
        scale = 1.0 / sigma * np.sqrt(2.0)
        print (scale)
        l.weight.data *= self.expand_dims(scale, l.weight.ndimension())

        # init bias
        if l.bias is not None:
          l.bias.data.zero_()
        if l.bias is not None and self.init_bias:
          y = l(x)
          mu = self.mean(y, self.per_channel)
          l.bias.data -= mu

    def forward(self, x, *kargs, **kwargs):
        #return self.seq(x)
        skip_data = {}
        count = 0
        skips = self.params['structure']['skips']
        for l in self.layers:
            if isinstance(l, torch.nn.modules.conv.Conv2d):
                '''if self.do_dynamic_init.sum():
                    #self.dynamic_initialize(l, x)
                    self.do_dynamic_init = torch.nn.Parameter(torch.zeros(1, requires_grad=False))'''
                count += 1
                if count in skip_data or str(count) in skip_data:
                    #print ('Getting {} from the skip bank'.format(torch.mean(skip_data[count])))
                    #print (torch.mean(torch.abs(x)), torch.var(x))
                    #print (torch.mean(torch.abs(skip_data[count])), torch.var(skip_data[count]))
                    x += skip_data[count]

            if isinstance(l, self.params['act']['constructor']):
                if count in skips or str(count) in skips:
                    #print ('Saving {} to the skip bank'.format(torch.mean(x)))
                    skip_data[skips[count]] = x
            x = l(x)
        return x
