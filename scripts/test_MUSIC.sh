#!/bin/bash

OPTS=""
OPTS+="--mode test "
OPTS+="--list_test data/test_limited.csv "

# Models
OPTS+="--arch_sound unet7 "
OPTS+="--arch_synthesizer linear "
OPTS+="--arch_frame resnet18dilated "
OPTS+="--img_pool maxpool_t "
OPTS+="--num_channels 32 "

# Binary mask / loss / log-freq / mix
OPTS+="--binary_mask 1 "
OPTS+="--loss bce "
OPTS+="--weighted_loss 1 "
OPTS+="--num_mix 2 "
OPTS+="--log_freq 1 "

# Frames
OPTS+="--num_frames 3 "
OPTS+="--stride_frames 24 "
OPTS+="--frameRate 8 "

# Audio
OPTS+="--audLen 65535 "
OPTS+="--audRate 11025 "

# Runtime
OPTS+="--batch_size_per_gpu 1 "
OPTS+="--num_gpus 1 "

python -u main.py $OPTS
