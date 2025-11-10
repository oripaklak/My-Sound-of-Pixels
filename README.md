# My Sound of Pixels

Sound-of-Pixels separates a musical mixture into its constituent instruments by learning a joint audio-visual representation on the [MUSIC dataset](https://sound-of-pixels.csail.mit.edu/). This repo contains the training/evaluation code (PyTorch), dataset utilities, visualization helpers, and an HTML demo generator for showcasing pixel-level audio predictions.

## Quick Start

```bash
git clone <your-fork-url>
cd My-Sound-of-Pixels
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision torchaudio numpy scipy imageio mir_eval wandb tqdm matplotlib soundfile
```

> **System requirements**: Python 3.8+, CUDA-capable GPU with ≥11 GB memory recommended, and `ffmpeg` available on the PATH for audio/video processing.

## Repository Layout

```
My-Sound-of-Pixels/
├── arguments.py          # CLI arguments shared by train/test
├── dataset/              # MUSIC dataset wrappers + video transforms
├── models/               # UNet (sound), ResNet (frames), synthesizer heads
├── scripts/              # Training / testing helper scripts
├── utils.py, viz.py      # STFT helpers, logging, HTML visualizer
├── ckpt/                 # Checkpoints, visualizations, demo artifacts
└── data/*.csv            # Lists of audio/video pairs used for each split
```

## Prepare the MUSIC Dataset

1. Download the official MUSIC dataset (audio-waveform pairs and extracted frames). Place it alongside this repo, e.g.
   ```
   project/
   ├── MUSIC_dataset/
   │   └── data/duet/{audio,frames}/<instrument pair>/<youtube_id>.{mp3,mp4}
   └── My-Sound-of-Pixels/
   ```
2. Each CSV in `data/` (train/val/test) is a comma-separated line:
   ```text
   ../MUSIC_dataset/data/duet/audio/flute violin/RWP6BHh_c7Y.mp3,../MUSIC_dataset/data/duet/frames/flute violin/RWP6BHh_c7Y.mp4,884
   ```
   - Column 1: path to the mixture audio clip  
   - Column 2: path to the directory containing extracted frames (`000001.jpg`, …)  
   - Column 3: frame count for the clip

Generate your own splits by mirroring the format above. Paths may be relative (recommended) or absolute.

## Training

The easiest way to launch training is via the helper script:

```bash
cd My-Sound-of-Pixels
bash scripts/train_MUSIC.sh
```

Key defaults (see `scripts/train_MUSIC.sh` and `arguments.py` for full list):

- Model: UNet7 audio separator + dilated ResNet18 frame encoder + linear synthesizer
- Mixes two instruments (`--num_mix 2`) with log-frequency STFTs
- 3 video frames sampled every 24 ticks at 8 fps
- Adam optimizers (`lr_sound 1e-3`, etc.) trained for 100 epochs with LR drops at epochs 40/80

Use either environment variables or in-script edits to customize dataset lists, learning rates, batch size, or checkpoint directory (`--ckpt`). Training logs and intermediate HTML visualizations are dumped under `ckpt/<run-id>/`.

## Testing / Evaluation

Generate predictions and separation metrics against a CSV split using:

```bash
bash scripts/test_MUSIC.sh
```

Important flags (see `scripts/test_MUSIC.sh`):

- `--mode test` forces deterministic center-frame sampling
- `--list_test data/test_limited.csv` selects the evaluation list
- `--binary_mask`, `--weighted_loss`, `--log_freq`, and other switches mirror the training configuration

Results are written to `ckpt/<id>/test/…`, including:

- `pred audio/pred{row}x{col}.wav`: pixel-level predictions on a spatial grid
- `pred masks/`, `pred heat map/`, `pixel activity/`: visual overlays for diagnostics
- Per-sample metadata in `meta.json`

Visual metrics (SDR/SIR/SAR) are printed to stdout and logged into the checkpoint directory.

## Visualization Assets

- `ckpt/visualization/<instrumentA>-<idA>+<instrumentB>-<idB>/` holds per-sample PNGs, audio snippets, and `meta.json` describing the prediction grid.
- `viz.py` provides helper utilities to compile those results into lightweight HTML reports (`HTMLVisualizer`).

## Tips & Troubleshooting

- **Custom checkpoints**: pass `--weights_sound`, `--weights_frame`, or `--weights_synthesizer` to `main.py` (see `arguments.py`) to fine-tune from pre-trained weights.
- **Mixed precision / multi-GPU**: the current scripts assume full precision and `--num_gpus 1`. Extend them if you need DistributedDataParallel.
- **wandb logging**: `main.py` imports `wandb`. Set `WANDB_MODE=offline` if you do not want to sync runs.
- **Missing dependencies**: `mir_eval` and `imageio` are required for evaluation and saving visualizations. Install them via `pip install mir_eval imageio`.

Feel free to adapt the scripts for new datasets or to experiment with different architectures—the argument parser exposes every knob used in the original Sound-of-Pixels paper.
