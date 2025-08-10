import torch
import torch.nn as nn
import torch.nn.functional as F

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image


class Resnet(nn.Module):
    def __init__(self, original_resnet):
        super(Resnet, self).__init__()
        self.features = nn.Sequential(
            *list(original_resnet.children())[:-1]) # remove the last layer - FC 
        # for param in self.features.parameters():
        # 	param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), x.size(1)) # x.size(0) - batch size, x.size(1) - feature channels (512)
        return x


class ResnetFC(nn.Module):
    def __init__(self, original_resnet, fc_dim=64,
                 pool_type='maxpool', conv_size=3):
        super(ResnetFC, self).__init__()
        self.pool_type = pool_type

        self.features = nn.Sequential(
            *list(original_resnet.children())[:-2]) # remove both FC layar and AvgPooling 

        self.fc = nn.Conv2d(
            512, fc_dim, kernel_size=conv_size, padding=conv_size//2) # Create a new FC layer

    def forward(self, x, pool=True):
        x = self.features(x)
        x = self.fc(x)

        if not pool:
            return x
        
        # This reduces the spatial dimensions (height and width) to 1x1 by taking the average\max of all pixels
        # The result will have the shape [batch_size, num_channels, 1, 1]
        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool2d(x, 1) 
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool2d(x, 1)
        x = x.view(x.size(0), x.size(1))
        return x

    def forward_multiframe(self, x, pool=True):
        (B, C, T, H, W) = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B*T, C, H, W)

        x = self.features(x)
        x = self.fc(x) # Conv2d

        (_, C, H, W) = x.size()
        x = x.view(B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        if not pool:
            return x

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool3d(x, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool3d(x, 1)

        x = x.view(B, C)
        return x


class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, fc_dim=64, pool_type='maxpool',
                 dilate_scale=16, conv_size=3):
        super(ResnetDilated, self).__init__()
        from functools import partial

        self.pool_type = pool_type

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16: 
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2)) # paramater change inside layer 4

        self.features = nn.Sequential(
            *list(orig_resnet.children())[:-2])
        
        # The K parameter in the paper corresponds to fc_dim
        self.fc = nn.Conv2d(
            512, fc_dim, kernel_size=conv_size, padding=conv_size//2)

    def _nostride_dilate(self, m, dilate):
        # remove the stride 
        classname = m.__class__.__name__
        # m is a layer in the model. e.g ResNet18
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, pool=True):
        x = self.features(x)
        x = self.fc(x)

        if not pool:
            return x

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool2d(x, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool2d(x, 1)

        x = x.view(x.size(0), x.size(1))
        return x

    def forward_multiframe(self, x, pool=True):
        (B, C, T, H, W) = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B*T, C, H, W)

        x = self.features(x) # (B*T, 512, H/16, W/16)
        x = self.fc(x) #(B*T, fc_dim, H/16, W/16) , fc_dim = args.num_channels

        (_, C, H, W) = x.size()
        x = x.view(B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        if not pool:
            return x
        # pool T, H, W
        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool3d(x, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool3d(x, 1)
        # pool T only
        elif self.pool_type == 'avgpool_t':
            x = F.adaptive_avg_pool3d(x, (1, H, W))  
        elif self.pool_type == 'maxpool_t':
            x = F.adaptive_max_pool3d(x, (1, H, W))

        if self.pool_type in ['avgpool_t', 'maxpool_t']:
            # shape: (B, C, 1, H, W)
            x = x.squeeze(2)  # remove the T dimension
            return x  # (B, C, H, W)   
                
        x = x.view(B, C)
        return x #(B, C)

