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
        B, C = sound_size[0], sound_size[1] 
        feat_img = feat_img.view(B, 1, C)   
        z = torch.bmm(feat_img * self.scale, feat_sound.view(B, C, -1)) \
            .view(B, 1, *sound_size[2:]) 
        z = z + self.bias
        return z

    def forward_nosum(self, feat_img, feat_sound):
        (B, C, H, W) = feat_sound.size()
        feat_img = feat_img.view(B, C)
        z = (feat_img * self.scale).view(B, C, 1, 1) * feat_sound 
        z = z + self.bias
        return z

    # inference purposes
    def forward_pixelwise(self, feats_img, feat_sound, spec_chunk=4096):
        (B, C, HI, WI) = feats_img.size()
        (_, _, HS, WS) = feat_sound.size()
        spatial = HI * WI
        spectral = HS * WS

        feats_img = feats_img.view(B, C, spatial).permute(0, 2, 1).contiguous()  # (B, spatial, C)
        feat_sound = feat_sound.view(B, C, spectral).contiguous()               # (B, C, spectral)

        scale = self.scale.view(1, 1, C)
        scaled_feats = feats_img * scale

        output_cpu = torch.empty(B, spatial, spectral,
                                 dtype=scaled_feats.dtype,
                                 device='cpu')

        chunk = max(1, int(spec_chunk))
        for start in range(0, spectral, chunk):
            end = min(start + chunk, spectral)
            sound_chunk = feat_sound[:, :, start:end]
            chunk_out = torch.bmm(scaled_feats, sound_chunk)
            output_cpu[:, :, start:end].copy_(chunk_out.detach().cpu())
            del chunk_out, sound_chunk

        output_cpu += self.bias.detach().cpu()
        del scaled_feats, feats_img, feat_sound
        output_cpu = output_cpu.view(B, HI, WI, HS, WS)
        return output_cpu
    
    def forward_pixelwise_multiframes(self, feats_img, feat_sound):
        (B, C, T, HI, WI) = feats_img.size()
        (B2, C2, HS, WS) = feat_sound.size()
        assert B == B2 and C == C2, "Batch and channel dimensions must match"

        feats_img_flat = feats_img.view(B, C, T, HI * WI).permute(0, 2, 3, 1)  # (B, T, HI*WI, C)
        feat_sound_flat = feat_sound.view(B, C, HS * WS)  # (B, C, HS*WS)

        # We need to multiply feats_img_flat (B, T, HI*WI, C) with feat_sound_flat (B, C, HS*WS)
        # Use einsum: for each t
        z = torch.einsum('btic, bcs -> btis', feats_img_flat, feat_sound_flat)
        z = z.view(B, T, HI, WI, HS, WS)
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
    def forward_pixelwise(self, feats_img, feat_sound, spec_chunk=4096):
        (B, C, HI, WI) = feats_img.size()
        (_, _, HS, WS) = feat_sound.size()
        spatial = HI * WI
        spectral = HS * WS

        feats_img = feats_img.view(B, C, spatial).permute(0, 2, 1).contiguous()
        feat_sound = feat_sound.view(B, C, spectral).contiguous()

        output_cpu = torch.empty(B, spatial, spectral,
                                 dtype=feats_img.dtype,
                                 device='cpu')
        chunk = max(1, int(spec_chunk))
        for start in range(0, spectral, chunk):
            end = min(start + chunk, spectral)
            sound_chunk = feat_sound[:, :, start:end]
            chunk_out = torch.bmm(feats_img, sound_chunk)
            output_cpu[:, :, start:end].copy_(chunk_out.detach().cpu())
            del chunk_out, sound_chunk

        output_cpu += self.bias.detach().cpu()
        del feats_img, feat_sound
        output_cpu = output_cpu.view(B, HI, WI, HS, WS)
        return output_cpu
    

