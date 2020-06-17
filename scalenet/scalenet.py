import numpy as np
import random
import os
import six
import copy
from pdb import set_trace as st

import torch

from .residuals import res_warp_img, res_warp_res, combine_residuals, downsample
from .model import Model
from .add import Add

class ScaleNet(Model):
    def __init__(self, max_level=14, name='pyramid', params={}):
        super().__init__()

        self.name = name
        self.params = params
        self.max_level = max_level

        self.level_upmodules = torch.nn.ModuleDict()
        self.level_downmodules = torch.nn.ModuleDict()
        self.level_uplinks = torch.nn.ModuleDict()
        self.level_downlinks = torch.nn.ModuleDict()
        self.level_skiplinks = torch.nn.ModuleDict()
        self.level_combiners = torch.nn.ModuleDict()

        for i in range(self.max_level):
            self.level_uplinks[str(i)]   = self.default_uplink()
            self.level_downlinks[str(i)] = self.default_downlink()
            self.level_skiplinks[str(i)] = self.default_skiplink()
            self.level_combiners[str(i)] = self.default_combiner()

    def forward(self, x, in_field=None, level_in=0, multimip_input=False):
        state = {}
        state = self.up_path(x, level_in, multimip_input, state)
        result = self.down_path(state, in_field=in_field)
        if True or 'debug' in self.params and self.params['debug']:
            self.state = state
        return result


    def up_path(self, x_in, level_in, multimip_input, state):
        state['up'] = {}
        all_module_levels = list(self.level_upmodules.keys()) + list(self.level_downmodules.keys())
        max_level = max([int(i) for i in all_module_levels])

        x = None
        for level in range(level_in, max_level + 1):
            level_state = {}
            state['up'][str(level)] = level_state

            if multimip_input and level in x_in:
                x = x_in[level]
            elif x is None:
                assert isinstance(x_in, torch.Tensor)
                x = x_in


            level_state['input'] = x
            if str(level) in self.level_upmodules:
                x = self.level_upmodules[str(level)](x, state, level)

            level_state['output'] = x

            skip = self.level_skiplinks[str(level)](x)
            level_state['skip'] = skip

            x = self.level_uplinks[str(level)](x)
            level_state['uplink'] = x

        return state

    def down_path(self, state, in_field=None):
        state['down'] = {}
        prev_out = None
        max_level = max([int(i) for i in state['up'].keys()])
        min_level = min([int(i) for i in state['up'].keys()])
        for level in reversed(range(min_level, max_level + 1)):

            level_state = {}
            state['down'][str(level)] = level_state

            skip = state['up'][str(level)]['skip']
            if prev_out is None:
                if in_field is not None:
                    prev_out = in_field
                    level_in = in_field
                    while level_in.shape[-1] > skip.shape[-1]:
                        level_in = downsample(level_in, is_res=True)
                else:
                    level_in = torch.zeros((skip.shape[0], 2, skip.shape[-2], skip.shape[-1]),
                        device=skip.device)
            else:
                downlink = self.level_downlinks[str(level)]
                level_in = downlink(prev_out)

            level_state['input'] = level_in
            if level in []:
                print ("SKIPPING LEVEL {}".format(level))
                level_out = level_in
            elif str(level) in self.level_downmodules:
                if prev_out is None:
                    downmodule_in = skip
                else:
                    downmodule_in = self.level_combiners[str(level)](skip, level_in, state, level)
                level_out = self.level_downmodules[str(level)](downmodule_in, state, level)
                if 'debug' in self.params and self.params['debug']:
                    print ("{}: mean {}, max {}".format(level, torch.mean(torch.abs(level_out)), level_out.abs().max()))
            else:
                level_out = level_in

            level_state['output'] = level_out
            prev_out = level_out


        result = state['down'][str(min_level)]['output']
        return result

    ############# Defaults ###############

    def default_uplink(self):
        return torch.nn.AvgPool2d(2, count_include_pad=False)

    def default_downlink(self):
        return torch.nn.Upsample(scale_factor=2, mode='bilinear')

    def default_combiner(self):
        return Add()

    def default_skiplink(self):
        return torch.nn.Sequential() #aka identity

    ############# Setters ###############

    def set_upmodule(self, module, level):
        self.level_upmodules[str(level)] = module

    def set_downmodule(self, module, level):
        self.level_downmodules[str(level)] = module

    def set_uplink(self, link, level):
        self.level_uplinks[str(level)] = link
        if len(list(link.parameters())) > 0:
            print ("Uplink mean weight [0] == {}".format(torch.mean(list(link.parameters())[0])))

    def set_downlink(self, link, level):
        self.level_downlinks[str(level)] = link

    def set_skiplink(self, link, level):
        self.level_skiplinks[str(level)] = link

    def set_combiner(self, combiner, level):
        self.level_combiners[str(level)] = combiner

    def set_all_uplinks(self, link):
        for level in range(self.max_level):
            self.level_uplinks[str(level)] = copy.deepcopy(link)

    def set_all_downlinks(self, link):
        for level in range(self.max_level):
            self.level_downlinks[str(level)] = copy.deepcopy(link)

    def set_all_skiplinks(self, link):
        for level in range(self.max_level):
            self.level_skiplinks[str(level)] = copy.deepcopy(link)

    def set_all_combiners(self, combiner):
        for level in range(self.max_level):
            self.level_combiners[str(level)] = copy.deepcopy(combiner)


    ############# Unsetters ###############
    def unset_upmodule(self, level):
        del self.level_upmodules[level]

    def unset_downmodule(self, level):
        del self.level_downmodules[level]

    def unset_uplink(self, level):
        del self.level_uplinks[level]

    def unset_downlink(self, level):
        del self.level_downlinks[level]

    def unset_skiplink(self, level):
        del self.level_skiplinks[level]

    def unset_combiner(self, level):
        del self.level_combiners[level]

