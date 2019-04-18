import numpy as np
import random
import os
import six
import copy

import torch

from .residuals import res_warp_img, res_warp_res, combine_residuals
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

    def forward(self, x, level_in):
        up_result = self.up_path(x, level_in)
        result = self.down_path(up_result)
        return result


    def up_path(self, x, level_in):
        result = {}
        all_module_levels = list(self.level_upmodules.keys()) + list(self.level_downmodules.keys())
        max_level = max([int(i) for i in all_module_levels])

        for level in range(level_in, max_level + 1):
            result[str(level)] = {}
            result[str(level)]['input'] = x
            if str(level) in self.level_upmodules:
                x = self.level_upmodules[str(level)](x)

            result[str(level)]['output'] = x

            skip = self.level_skiplinks[str(level)](x)
            result[str(level)]['skip'] = skip

            x = self.level_uplinks[str(level)](x)
            result[str(level)]['uplink'] = x

        return result

    def down_path(self, up_result):
        prev_out = None
        max_level = max([int(i) for i in up_result.keys()])
        min_level = min([int(i) for i in up_result.keys()])

        for level in reversed(range(min_level, max_level + 1)):
            skip = up_result[str(level)]['skip']
            if prev_out is None:
                prev_out = torch.zeros_like(skip, device=skip.get_device())
            if str(level) in self.level_downmodules:
                downmodule_in = self.level_combiners[str(level)](skip, prev_out, level, up_result)
                out = self.level_downmodules[str(level)](downmodule_in)
            else:
                out = prev_out

            prev_out = out

            if level != min_level:
                prev_out = self.level_downlinks[str(level)](out)

        return prev_out

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

