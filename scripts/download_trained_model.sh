#!/bin/bash

# Set model ID and path
MODEL_ID="MUSIC-2mix-LogFreq-resnet18dilated-unet7-linear-frames3stride24-maxpool-binary-weightedLoss-channels32-epoch100-step40_80"
MODEL_PATH="./ckpt/${MODEL_ID}"
LINK_RELEASE="http://sound-of-pixels.csail.mit.edu/release"

# Define file URLs
LIST_VAL="${LINK_RELEASE}/val.csv"
WEIGHTS_FRAME="${LINK_RELEASE}/${MODEL_ID}/frame_best.pth"
WEIGHTS_SOUND="${LINK_RELEASE}/${MODEL_ID}/sound_best.pth"
WEIGHTS_SYNTHESIZER="${LINK_RELEASE}/${MODEL_ID}/synthesizer_best.pth"

# Create data directory if it doesn't exist
mkdir -p ./data

# Download validation CSV
wget -O ./data/val.csv "$LIST_VAL"

# Create model checkpoint directory if it doesn't exist
mkdir -p "$MODEL_PATH"

# Download model weights
wget -O "${MODEL_PATH}/sound_best.pth" "$WEIGHTS_SOUND"
wget -O "${MODEL_PATH}/frame_best.pth" "$WEIGHTS_FRAME"
wget -O "${MODEL_PATH}/synthesizer_best.pth" "$WEIGHTS_SYNTHESIZER"
