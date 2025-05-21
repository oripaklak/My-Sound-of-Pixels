import torch
import torch.nn as nn
import torch.nn.functional as F

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from my_dataset.save_image import save_image # to save the image 



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

        x = self.features(x)
        x = self.fc(x)

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

# Main function for testing the classes
def main():
    # Parameters to load and save the image 
    directory = "C:/Users/User/OneDrive/BIU/project/Sound-of-Pixels/my_dataset"
    base_name = "guitar"
    extension = "jpg"
    image_path =  directory + "/" + base_name + "." + extension

    # Load a pretrained ResNet model
    from torchvision.models import ResNet18_Weights
    original_resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # Load a real image from the file system
    image = Image.open(image_path)

    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),          # Resize the image to 224x224
        transforms.ToTensor(),                  # Convert to a PyTorch tensor
        transforms.Normalize(                   # Normalize the image
            mean=[0.485, 0.456, 0.406],         # Mean for ImageNet - same as base.py
            std=[0.229, 0.224, 0.225]           # Std for ImageNet - same as base.py
        )
    ])
    
    # Add a batch dimension: [1, C, H, W]
    input_tensor = preprocess(image).unsqueeze(0)  

    # Create an instance of ResnetFC
    resnet_fc = ResnetFC(original_resnet, fc_dim=16, pool_type='avgpool')

    # Run the model on the image
    output_fc = resnet_fc(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape (ResnetFC): {output_fc.shape}")

    # Create an instance of ResnetDilated
    resnet_dilated = ResnetDilated(original_resnet, fc_dim=16, dilate_scale=8)
    output_dilated = resnet_dilated(input_tensor)
    print(f"Output shape (ResnetDilated): {output_dilated.shape}")

    # Plot the output features as a heatmap
    output_fc_reshaped = output_dilated.detach().numpy().reshape(-1, 1)  # Assuming a single vector output
    fig,ax = plt.subplots(figsize=(10, 2))
    cax = ax.imshow(output_fc_reshaped, cmap="viridis", aspect="auto")
    fig.colorbar(cax)
    ax.set_title("Model Output Features")

    # Save the image
    unique_path = save_image(directory, base_name, extension) # Generate a unique file path
    fig.savefig(unique_path)
    plt.close(fig)


if __name__ == "__main__":
    main()