import torch
import torch.nn as nn
import torch.nn.functional as F

# libraries i added 
import librosa
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from dataset.base import BaseDataset
from my_dataset.save_image import save_image # to save the image 

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
        

# Main function for testing the classes
def main():

    # Parameters to load and save the image 
    directory = "C:/Users/User/OneDrive/BIU/project/Sound-of-Pixels/my_dataset"
    base_name = "guitar"
    extension = "wav"
    audio_path =  directory + "/" + base_name + "." + extension

    # Parameters related to the paper
    audLen = 65535      # Target audio length (in samples)
    audRate = 11025     # sampling rate
    duration = 6        # seconds
    samples_to_keep = audRate * duration
    stft_frame = 1022   # window size
    stft_hop = 256      # hop lenght
    center_timestamp = 30 # seconds 
    log_freq = True 

    # Load the audio file
    audio_raw, audRate = librosa.load(audio_path, sr=audRate)

    # Crop or pad the audio to match audLen
    len_raw = audio_raw.shape[0]
    center = int(center_timestamp * audRate)  # Convert center timestamp to sample index
    start = max(0, center - audLen // 2)
    end = min(len_raw, center + audLen // 2)

    # Initialize the cropped audio with zeros (to handle padding if needed)
    audio_cropped = np.zeros(audLen, dtype=audio_raw.dtype)

    # Copy the valid portion of the audio into the center of the target array
    crop_start = audLen // 2 - (center - start)
    crop_end = audLen // 2 + (end - center)
    audio_cropped[crop_start:crop_end] = audio_raw[start:end]

    # Replace `audio_raw` with the cropped version
    audio_raw = audio_cropped
    print("audio length: ", len(audio_raw))
 
    # Prepare a mock "opt" parameter to simulate BaseDataset behavior
    class Options:
        def __init__(self):
            self.num_frames = None
            self.stride_frames = None
            self.audRate = audRate
            self.audLen = audLen
            self.stft_frame = stft_frame
            self.stft_hop = stft_hop
            self.frameRate = None
            self.imgSize = 224
            self.binary_mask = None
            self.log_freq = log_freq
            self.seed = None
            self.dup_trainset = 100
            #self.log_freq = True  # Enable log frequency scaling

    opt = Options()

    # Create a BaseDataset instance
    dataset = BaseDataset([audio_path], opt)

    # (512, 256): 512 = ceil(1022 / 512) , 256 = ceil(65535 / 256)
    # Perform STFT using BaseDataset
    amp, phase = dataset._stft(audio_raw)
    print(f"STFT Matrix Shape: {amp.shape}")  


    if log_freq:
        # Convert linear frequency bins to log-frequency bins
        mel_filter = librosa.filters.mel(
            sr=audRate,
            n_fft=stft_frame,
            n_mels=256  # Target 256 frequency bins
        )
        amp = np.dot(mel_filter, amp)

    print(f"STFT Matrix Shape: {amp.shape}")  # (256, 256)

    # --------------- UNET ------------------
    # Convert amp to a 4D tensor with batch and channel dimensions + Convert to PyTorch tensor
    amp = torch.from_numpy(amp).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 256, 256)

    # Initialize the Unet model
    unet = Unet(fc_dim=16, num_downs=5, ngf=64, use_dropout=False)

    # Run the amp tensor through the Unet
    output = unet(amp)

    # Print the output shape
    print(f"Unet Input Shape: {amp.shape}")
    print(f"Unet Output Shape: {output.shape}")

    # Plot 
    # Remove the batch dimension (shape becomes (16, 256, 256))
    
    spectrograms = output.squeeze(0)

    # Create a 4x4 grid to display all 16 spectrograms
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        # Convert the spectrogram to NumPy (if it's not already)
        spectrogram = spectrograms[i].detach().numpy()  # Ensure it's on CPU and NumPy array
        
        # Plot the spectrogram
        ax.imshow(spectrogram, aspect='auto', origin='lower', cmap='jet')
        ax.set_title(f"Channel {i+1}")
        ax.axis("off")  # Hide axes for cleaner visualization

    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()


