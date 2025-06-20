import torch
import torch.nn as nn
import torch.nn.functional as F

# libraries i added 
import librosa
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from dataset.base import BaseDataset

# ngf = number of generator filters 
# The K parameter in the paper corresponds to fc_dim (same as vision_net)
class Unet(nn.Module):
    def __init__(self, fc_dim=64, num_downs=5, ngf=64, use_dropout=False):
        super(Unet, self).__init__()

        # construct unet structure
        unet_block = UnetBlock(
            ngf * 8, ngf * 8, input_nc=None,
            submodule=None, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetBlock(
                ngf * 8, ngf * 8, input_nc=None,
                submodule=unet_block, use_dropout=use_dropout)
        unet_block = UnetBlock(
            ngf * 4, ngf * 8, input_nc=None,
            submodule=unet_block)
        unet_block = UnetBlock(
            ngf * 2, ngf * 4, input_nc=None,
            submodule=unet_block)
        unet_block = UnetBlock(
            ngf, ngf * 2, input_nc=None,
            submodule=unet_block)
        unet_block = UnetBlock(
            fc_dim, ngf, input_nc=1,
            submodule=unet_block, outermost=True)

        self.bn0 = nn.BatchNorm2d(1) # 1 stands for 1 channel
        self.unet_block = unet_block

    def forward(self, x):
        x = self.bn0(x)
        x = self.unet_block(x)
        return x


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetBlock(nn.Module):
    def __init__(self, outer_nc, inner_input_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False,
                 use_dropout=False, inner_output_nc=None, noskip=False):
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        self.noskip = noskip
        use_bias = False

        if input_nc is None: # defined on all blocks except for the most outer 
            input_nc = outer_nc
        if innermost: 
            inner_output_nc = inner_input_nc
        elif inner_output_nc is None:
            inner_output_nc = 2 * inner_input_nc

        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(inner_input_nc)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(outer_nc)
        upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True) 
        # bilinear for 3D input Tensor (e.g, minibatch X channels X optional depth)
        # input: (N, C, W) -> output: (N, C, W x scale_factor)

        if outermost:
            # (N, input_nc, W, H) -> (N, inner_input_nc, W/2, H/2)
            downconv = nn.Conv2d(
                input_nc, inner_input_nc, kernel_size=4,
                stride=2, padding=1, bias=use_bias)
            # (N, inner_output_nc, W, H) -> (N, outer_nc, W, H)
            upconv = nn.Conv2d(
                inner_output_nc, outer_nc, kernel_size=3, padding=1)

            down = [downconv]
            up = [uprelu, upsample, upconv] # no upnorm
            model = down + [submodule] + up

        elif innermost:
            # (N, input_nc, W, H) -> (N, inner_input_nc, W/2, H/2)
            downconv = nn.Conv2d(
                input_nc, inner_input_nc, kernel_size=4,
                stride=2, padding=1, bias=use_bias)
            # (N, inner_output_nc, W, H) -> (N, outer_nc, W, H)           
            upconv = nn.Conv2d(
                inner_output_nc, outer_nc, kernel_size=3,
                padding=1, bias=use_bias)

            down = [downrelu, downconv] # no dowmnorm
            up = [uprelu, upsample, upconv, upnorm]
            model = down + up # no submodule

        else: # intermidiate submodules 
            # (N, input_nc, W, H) -> (N, inner_input_nc, W/2, H/2)
            downconv = nn.Conv2d(
                input_nc, inner_input_nc, kernel_size=4,
                stride=2, padding=1, bias=use_bias)
            # (N, inner_output_nc, W, H) -> (N, outer_nc, W, H)
            upconv = nn.Conv2d(
                inner_output_nc, outer_nc, kernel_size=3,
                padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upsample, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost or self.noskip:
            return self.model(x)
        else: # with skip!
            return torch.cat([x, self.model(x)], 1) # x concatenated with model(x) horizontally
        
