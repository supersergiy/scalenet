import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
from six import iteritems
import os

from .residuals import res_warp_img, res_warp_res, combine_residuals
from .model import Model

class ScaleNet(Model):
    def __init__(self, max_mip=14, name='pyramid', params={}):
        super().__init__()

        self.name = name
        self.params = params
        self.max_mip = max_mip

        self.mip_upmodules = torch.nn.ModuleDict()
        self.mip_downmodules = torch.nn.ModuleDict()
        self.mip_uplinks = torch.nn.ModuleDict()
        self.mip_downlinks = torch.nn.ModuleDict()
        self.mip_skiplinks = torch.nn.ModuleDict()
        self.mip_combiners = torch.nn.ModuleDict()

        for i in range(self.max_mip):
            self.mip_uplinks[str(i)]   = self.default_uplink()
            self.mip_downlinks[str(i)] = self.default_downlink()
            self.mip_skiplinks[str(i)] = self.default_skiplink()
            self.mip_combiners[str(i)] = self.default_combiner()

    def forward(self, x, mip_in):
        up_result = self.up_path(x, mip_in)
        result = self.down_path(up_result)
        return result


    def up_path(self, x, mip_in):
        result = {}
        all_module_mips = list(self.mip_upmodules.keys()) + list(self.mip_downmodules.keys())
        max_mip = max([int(i) for i in all_module_mips])

        for mip in range(mip_in, max_mip + 1):
            result[str(mip)] = {}
            result[str(mip)]['input'] = x
            if str(mip) in self.mip_upmodules:
                x = self.mip_upmodules[str(mip)](x)

            result[str(mip)]['output'] = x

            skip = self.mip_skiplinks[str(mip)](x)
            result[str(mip)]['skip'] = skip

            x = self.mip_uplinks[str(mip)](x)
            result[str(mip)]['uplink'] = x

        return result

    def down_path(self, up_result):
        prev_out = None
        max_mip = max([int(i) for i in up_result.keys()])
        min_mip = min([int(i) for i in up_result.keys()])

        for mip in reversed(range(min_mip, max_mip + 1)):
            skip = up_result[str(mip)]['skip']
            if prev_out is None:
                prev_out = torch.zeros_like(skip, device=skip.get_device())
            if str(mip) in self.mip_downmodules:
                downmodule_in = self.mip_combiners[str(mip)](skip, prev_out)
                out = self.mip_downmodules[str(mip)](downmodule_in)
            else:
                out = prev_out


            prev_out = out

            if mip != min_mip:
                prev_out = self.mip_downlinks[str(mip)](out)

        return prev_out

    def compute_downsamples(self, img, curr_mip, max_mip):
        downsampled_src_tgt = {}
        downsampled_src_tgt[curr_mip] = img
        #print ("input shape: {}".format(downsampled_src_tgt[curr_mip].shape))
        for mip in range(curr_mip + 1, max_mip + 1):
            # logger.debug("Downsampl MIP {}".format(mip))
            downsampled_src_tgt[mip] = self.mip_downsamplers[mip](
                                                downsampled_src_tgt[mip - 1])
            #print ("downsample {} shape: {}".format(mip, downsampled_src_tgt[mip].shape))
        return downsampled_src_tgt


    def get_downsampled_src(self, downsampled_src_tgt, mip):
        emb_dims = downsampled_src_tgt[mip].shape[1]
        src = downsampled_src_tgt[mip][:, 0:emb_dims//2]
        return src

    def get_downsampled_tgt(self, downsampled_src_tgt, mip):
        emb_dims = downsampled_src_tgt[mip].shape[1]
        tgt = downsampled_src_tgt[mip][:, emb_dims//2:]
        return tgt

    '''def forward(self, src_tgt, mip_in):
        src_tgt_var = Variable(src_tgt.cuda(), requires_grad=True)
        # Find which mips are to be applied
        mips_to_process = iteritems(self.mip_processors)
        filtered_mips   = [(int(m[0]), m[1]) for m in mips_to_process if int(m[0]) >= mip_in]
        ordered_mips    = list(reversed(sorted(filtered_mips)))

        high_mip = ordered_mips[0][0]
        low_mip  = ordered_mips[-1][0]

        # Set up auxilary structures
        residuals  = {}
        downsampled_src_tgt = self.compute_downsamples(src_tgt_var,
                                                       mip_in,
                                                       high_mip)
        # logger.debug("Setting downsample MIP {}".format(high_mip))
        aggregate_res = None
        aggregate_res_mip = None
        # Goal of this loop is to compute aggregate_res
        for mip, module in ordered_mips:
            # generate a warped image at $mip
            if aggregate_res is not None:
                while aggregate_res_mip > mip:
                    aggregate_res = self.upsample_residuals(aggregate_res)
                    aggregate_res_mip -= 1
                if 'rollback' in self.params and self.params['rollback']:
                    tmp_aggregate_res = aggregate_res
                    for i in range(mip_in, aggregate_res_mip):
                        tmp_aggregate_res = self.upsample_residuals(tmp_aggregate_res)

                    raw_src_tgt = downsampled_src_tgt[mip_in] #self.get_downsampled_src(downsampled_src_tgt, mip_in)
                    src_tgt = res_warp_img(
                                       raw_src_tgt,
                                       tmp_aggregate_res)
                    for m in range(mip_in, mip):
                        src_tgt = self.mip_downsamplers[m](src_tgt)
                    src = src_tgt[:, :src_tgt.shape[1]//2]
                else:
                    raw_src = self.get_downsampled_src(downsampled_src_tgt, mip)
                    src = res_warp_img(
                                       raw_src,
                                       aggregate_res)
            else:
                src = self.get_downsampled_src(downsampled_src_tgt, high_mip)

            # Compute residual at level $mip
            tgt = self.get_downsampled_tgt(downsampled_src_tgt, mip)
            sample = torch.cat((src, tgt), 1)
            residuals[mip] = self.mip_processors[mip](sample).permute(0, 2, 3, 1)
            if 'debug' in self.params and self.params['debug']:
                print ("Mip {:2d}: Average {:.2E} Max {:.2E}".format(mip,
                    torch.mean(torch.abs(residuals[mip][..., 4:-4, 4:-4, :])),
                    torch.max(torch.abs(residuals[mip][..., 4:-4, 4:-4, :]))))
            # initialize aggregate flow if None
            if aggregate_res is None:
                aggregate_res = torch.zeros(residuals[mip].shape, device='cuda')
                aggregate_res_mip = mip

            # Add the residual at level $mip to $aggregate_flow
            aggregate_res = combine_residuals(
                                              residuals[mip],
                                              aggregate_res,
                                              is_pix_res=True)

        while aggregate_res_mip > mip_in:
            aggregate_res = self.upsample_residuals(aggregate_res)
            aggregate_res_mip -= 1
        return aggregate_res'''
    ############# Defaults ###############

    def default_uplink(self):
        return torch.nn.AvgPool2d(2, count_include_pad=False)

    def default_downlink(self):
        return torch.nn.Upsample(scale_factor=2, mode='bilinear')

    def default_combiner(self):
        return AddModule()

    def default_skiplink(self):
        return nn.Sequential() #aka identity

    ############# Setters ###############

    def set_upmodule(self, module, mip):
        self.mip_upmodules[str(mip)] = module

    def set_downmodule(self, module, mip):
        self.mip_downmodules[str(mip)] = module

    def set_uplink(self, link, mip):
        self.mip_uplinks[str(mip)] = link

    def set_downlink(self, link, mip):
        self.mip_downlinks[str(mip)] = link

    def set_skiplink(self, link, mip):
        self.mip_skiplinks[str(mip)] = link

    def set_combiner(self, combiner, mip):
        self.mip_combiners[str(mip)] = combiner

    ############# Unsetters ###############
    def unset_upmodule(self, mip):
        del self.mip_upmodules[mip]

    def unset_downmodule(self, mip):
        del self.mip_downmodules[mip]

    def unset_uplink(self, mip):
        del self.mip_uplinks[mip]

    def unset_downlink(self, mip):
        del self.mip_downlinks[mip]

    def unset_skiplink(self, mip):
        del self.mip_skiplinks[mip]

    def unset_combiner(self, mip):
        del self.mip_combiners[mip]



class AddModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x + y
