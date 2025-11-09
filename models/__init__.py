import torch
import torchvision
import torch.nn.functional as F

from torchvision.models import resnet18, ResNet18_Weights

from .synthesizer_net import InnerProd, Bias
from .audio_net import Unet
from .vision_net import ResnetFC, ResnetDilated
from .criterion import BCELoss, L1Loss, L2Loss
from utils import  warpgrid

def activate(x, activation):
    if activation == 'sigmoid':
        return torch.sigmoid(x)
    elif activation == 'softmax':
        return F.softmax(x, dim=1)
    elif activation == 'relu':
        return F.relu(x)
    elif activation == 'tanh':
        return F.tanh(x)
    elif activation == 'no':
        return x
    else:
        raise Exception('Unkown activation!')


class ModelBuilder():
    # custom weights initialization
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.001)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.0001)

    def build_sound(self, arch='unet5', fc_dim=64, weights=''):
        # 2D models
        if arch == 'unet5':
            net_sound = Unet(fc_dim=fc_dim, num_downs=5)
        elif arch == 'unet6':
            net_sound = Unet(fc_dim=fc_dim, num_downs=6)
        elif arch == 'unet7':
            net_sound = Unet(fc_dim=fc_dim, num_downs=7)
        else:
            raise Exception('Architecture undefined!')

        net_sound.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_sound')
            net_sound.load_state_dict(torch.load(weights))

        return net_sound

    # builder for vision
    def build_frame(self, arch='resnet18', fc_dim=64, pool_type='avgpool', weights=''): 
        pretrained=True 
        if arch == 'resnet18fc': 
            original_resnet = torchvision.models.resnet18(pretrained) 
            net = ResnetFC( 
                original_resnet, fc_dim=fc_dim, pool_type=pool_type) 
        elif arch == 'resnet18dilated':
            original_resnet = torchvision.models.resnet18(pretrained) 
            net = ResnetDilated( 
                original_resnet, fc_dim=fc_dim, pool_type=pool_type) 
        else: 
            raise Exception('Architecture undefined!') 
        if len(weights) > 0: 
            print('Loading weights for net_frame') 
            net.load_state_dict(torch.load(weights)) 
        return net

    def build_synthesizer(self, arch, fc_dim=64, weights=''):
        if arch == 'linear':
            net = InnerProd(fc_dim=fc_dim)
        elif arch == 'bias':
            net = Bias()
        else:
            raise Exception('Architecture undefined!')

        net.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_synthesizer')
            net.load_state_dict(torch.load(weights))
        return net

    def build_criterion(self, arch):
        if arch == 'bce':
            net = BCELoss()
        elif arch == 'l1':
            net = L1Loss()
        elif arch == 'l2':
            net = L2Loss()
        else:
            raise Exception('Architecture undefined!')
        return net

class ModelTest(torch.nn.Module):
    def __init__(self,nets):
        super(ModelTest,self).__init__()
        self.net_sound, self.net_frame, self.net_synthesizer = nets

    def forward(self, batch_data , args):
        mag = batch_data["mag"]
        device = mag.device if mag.is_cuda else args.device
        if not mag.is_cuda:
            mag = mag.to(device, non_blocking=True)
        mag = mag.unsqueeze(1) # (B, 1, HS, WS)
        frames = batch_data["frames"]
        if not frames.is_cuda:
            frames = frames.to(device, non_blocking=True) # (B, C, T, HI, WI)
        B = mag.size(0)
        T = mag.size(3)
        
        if args.log_freq:
            grid_warp = torch.from_numpy(warpgrid(B, 256, T, warp=True)).to(device)
            mag = F.grid_sample(mag, grid_warp) # (B, 1, 256, 256)
            mag = torch.log(mag).detach()

        # 1. forward net_sound -> (B, C, HS, WS)
        feat_sound = self.net_sound(mag)
        feat_sound = activate(feat_sound, args.sound_activation)

        # 2. forward net_frame -> (B, C, HI/16, WI/16) -> resize to input resolution
        feat_frames = self.net_frame.forward_multiframe(frames)
        feat_frames = activate(feat_frames, args.img_activation)
        if feat_frames.dim() == 4:
            target_hw = int(getattr(args, 'imgSize', 224))
            feat_frames = F.interpolate(
                feat_frames,
                size=(target_hw, target_hw),
                mode='bilinear',
                align_corners=False)
        
        # 3. sound synthesizer -> (B, target_H, target_W, HS, WS)
        pred_masks = self.net_synthesizer.forward_pixelwise(feat_frames, feat_sound)
        pred_masks = activate(pred_masks, args.output_activation)

        # 4. return the audio
        return pred_masks
