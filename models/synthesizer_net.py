import torch
import torch.nn as nn
import torch.nn.functional as F


class InnerProd(nn.Module):
    def __init__(self, fc_dim):
        super(InnerProd, self).__init__()
        # used as a learnable parameter within a model
        self.scale = nn.Parameter(torch.ones(fc_dim)) 
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, feat_img, feat_sound):
        sound_size = feat_sound.size() 
        # C = fc_dim
        B, C = sound_size[0], sound_size[1] # audio_net output's shape is (B, C, 256, 256)
        feat_img = feat_img.view(B, 1, C)   # vision_net output's shape is (B, C) ;change dimensions to (B, 1, C) 
        # bmm = batch matrix-matrix
        # torch.bmm(input, mat2) -> (b,n,m), (b,m,p) -> output is tensor: (b,n,p)
        # feat_img * self.scale -> apply a learnable scaling factor to each feature channel
        # feat_sound.view(B, C, -1) -> (B, C, 65536)
        # z after torch.bmm -> (B, 1, 65536)
        # z after view -> (B, 1, 256, 256)
        z = torch.bmm(feat_img * self.scale, feat_sound.view(B, C, -1)) \
            .view(B, 1, *sound_size[2:]) 
        z = z + self.bias
        return z

    def forward_nosum(self, feat_img, feat_sound):
        (B, C, H, W) = feat_sound.size()
        feat_img = feat_img.view(B, C)
        z = (feat_img * self.scale).view(B, C, 1, 1) * feat_sound # z size: (B,C,256,256)
        z = z + self.bias
        return z

    # inference purposes
    def forward_pixelwise(self, feats_img, feat_sound):
        (B, C, HI, WI) = feats_img.size() # but the feats_img should be (B,C) - where is another 2 elements??
        (B, C, HS, WS) = feat_sound.size()
        feats_img = feats_img.view(B, C, HI*WI)
        feats_img = feats_img.transpose(1, 2) # output: (B, HI*WI, C)
        feat_sound = feat_sound.view(B, C, HS * WS)
        z = torch.bmm(feats_img * self.scale, feat_sound) \
            .view(B, HI, WI, HS, WS)
        z = z + self.bias
        return z


class Bias(nn.Module):
    def __init__(self):
        super(Bias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        # self.bias = nn.Parameter(-torch.ones(1))

    def forward(self, feat_img, feat_sound):
        (B, C, H, W) = feat_sound.size()
        feat_img = feat_img.view(B, 1, C)
        z = torch.bmm(feat_img, feat_sound.view(B, C, H * W)).view(B, 1, H, W)
        z = z + self.bias
        return z

    def forward_nosum(self, feat_img, feat_sound):
        (B, C, H, W) = feat_sound.size()
        z = feat_img.view(B, C, 1, 1) * feat_sound
        z = z + self.bias
        return z

    # inference purposes
    def forward_pixelwise(self, feats_img, feat_sound):
        (B, C, HI, WI) = feats_img.size()
        (B, C, HS, WS) = feat_sound.size()
        feats_img = feats_img.view(B, C, HI*WI)
        feats_img = feats_img.transpose(1, 2)
        feat_sound = feat_sound.view(B, C, HS * WS)
        z = torch.bmm(feats_img, feat_sound) \
            .view(B, HI, WI, HS, WS)
        z = z + self.bias
        return z
