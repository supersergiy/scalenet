from torch import nn
from torch.autograd import Variable
import numpy as np
import torch
import six
import os

from model import Model
from tools import dict_to_str

class Sequence(Model):
    def initc(self, m, mult):
        m.weight.data *= mult

    def validate_arch_desc(self, arch_desc):
        #TODO
        return

    def __init__(self, arch_desc):
        super().__init__()
        self.name = dist_to_str(arch_desc)
        self.params = parse_arch_desc(self, arch_desc)
        self.construct_layers(self.params)

    def parse_arch_desc(self, arch_desc):
        self.validate_arch_desc(arch_desc)
        params = {
            'flags': {
                "use_batchnorm": False,
                "use_inputnorm": False,
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

    def construct_layters(self, params):
        fms = params['structure']['fms']
        skips = params['structure']['skips']

        init_mult = params['conv']['init_mult']
        k = params['conv']['k']
        pad = (k - 1) // 2

        act_constructor = params['act']['constructor']
        self.layers = torch.nn.ModuleList()

        if params['flags']['use_inputnorm']:
            self.layers.append(torch.nn.InstanceNorm2d(num_features=fms[0]))

        for i in range(len(fms) - 1):
            self.layers.append(nn.Conv2d(fms[i], fms[i + 1], k, padding=p))
            self.initc(self.layers[-1], init_mult)

            if i != len(fms) - 2:
                if self.params['flags']['use_batchnorm']:
                    self.layers.append(nn.BatchNorm2d(fms[i + 1]))
                if self.params['flags']['use_instancenorm']:
                    self.layers.append(torch.nn.InstanceNorm2d(
                                                  num_features=fms[i + 1]))
                self.layers.append(act_constructor())

        self.seq = nn.Sequential(*self.layers)

    def parse_conv_params(self, params, arch_desc):
        if 'k' in arch_desc:
            params['conv']['k'] = arch_desc['k']

        if 'conv_init_mult' in arch_desc:
            params['conv']['init_mult'] = arch_desc['conv_init_mult']

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

    def parse_flags_params(self, params, arch_desc):
        if 'flags' in arch_desc and 'batchnorm' in arch_desc['flags']:
            params['flags']['use_batchnorm'] = True

        if 'flags' in arch_desc and 'instancenorm' in arch_desc['flags']:
            params['flags']['use_instancenorm'] = True

        if 'flags' in arch_desc and 'inputnorm' in arch_desc['flags']:
            params['flags']['use_inputnorm'] = True

    def parse_structure_params(self, params, arch_desc):
        params['structure']['fms'] = arch_desc['fms']
        if 'skips' in arch_desc:
            self.params['structure']['skips'] = {int(s): e for (s, e) in six.iteritems(arch_desc['skips'])}


    def forward(self, x):
        skip_data = {}
        count = 0

        for l in self.layers:
            if isinstance(l, torch.nn.modules.conv.Conv2d):
                count += 1
                if count in skip_data or str(count) in skip_data:
                    #print ('Getting {} from the skip bank'.format(torch.mean(skip_data[count])))
                    #print (torch.mean(torch.abs(x)), torch.var(x))
                    #print (torch.mean(torch.abs(skip_data[count])), torch.var(skip_data[count]))
                    x += skip_data[count]

            if isinstance(l, self.params['act_f']):
                if count in self.params['skips'] or str(count) in self.params['skips']:
                    #print ('Saving {} to the skip bank'.format(torch.mean(x)))
                    skip_data[self.params['skips'][count]] = x

            x = l(x)

        return x