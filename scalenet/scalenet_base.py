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

        self.best_val = 1000000
        self.name = name
        self.params = params

        self.mip_processors = {}
        self.mip_downsamplers = {}
        self.max_mip = max_mip

        for i in range(self.max_mip):
            self.mip_downsamplers[i] = self.default_downsampler()

    def default_downsampler(self):
        return nn.AvgPool2d(2, count_include_pad=False)

    def set_mip_processor(self, module, mip):
        self.mip_processors[mip] = module

    def unset_mip_processor(self, mip):
        del self.mip_processors[mip]

    def set_mip_downsampler(self, module, mip):
        self.mip_downsamplers[mip] = module

    def unset_mip_downsampler(self, mip):
        self.mip_downsamplers[mip] = self.default_downsampler()

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

    def get_all_processor_params(self):
        params = []
        for mip, module in iteritems(self.mip_processors):
            params.extend(module.parameters())
        return params

    def get_processor_params(self, mip):
        params = []
        params.extend(self.mip_processors[mip].parameters())
        return params


    def get_all_downsampler_params(self):
        params = []
        for mip, module in iteritems(self.mip_downsamplers):
            params.extend(module.parameters())
        return params

    def get_downsampler_params(self, mip):
        return [self.mip_downsamplers[mip].parameters()]

    def get_all_params(self):
        params = []
        params.extend(self.get_all_processor_params())
        params.extend(self.get_all_downsampler_params())
        return params

    def upsample_residuals(self, residuals):
        result = nn.functional.interpolate(residuals.permute(
                                         0, 3, 1, 2), scale_factor=2).permute(0, 2, 3, 1)
        result *= 2

        return result

    def get_downsampled_src(self, downsampled_src_tgt, mip):
        emb_dims = downsampled_src_tgt[mip].shape[1]
        src = downsampled_src_tgt[mip][:, 0:emb_dims//2]
        return src

    def get_downsampled_tgt(self, downsampled_src_tgt, mip):
        emb_dims = downsampled_src_tgt[mip].shape[1]
        tgt = downsampled_src_tgt[mip][:, emb_dims//2:]
        return tgt

    def forward(self, src_tgt, mip_in):
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
        return aggregate_res
