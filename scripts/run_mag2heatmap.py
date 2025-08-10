import matplotlib.pyplot as plt
import librosa
import torchaudio
import torch 
import numpy as np
import cv2

def magnitude2heatmap(mag, log=True, scale=200.):
    print("Raw mag min/max:", mag.min(), mag.max())
    if log:
        mag = np.log10(mag + 1.)
    print("After log10, min/max:", mag.min(), mag.max())
    mag *= scale
    mag[mag > 255] = 255
    mag = mag.astype(np.uint8)
    mag_color = cv2.applyColorMap(mag, cv2.COLORMAP_JET)
    mag_color = mag_color[:, :, ::-1]
    return mag_color

def load_audio_file(path):
    if path.endswith('.mp3'):
        audio_raw, rate = torchaudio.load(path)
        audio_raw = audio_raw.numpy().astype(np.float32)

        # range to [-1, 1]
        #audio_raw *= (2.0**-31)

        # convert to mono
        if audio_raw.shape[0] == 2:
            audio_raw = (audio_raw[0, :] + audio_raw[1, :]) / 2
        else:
            audio_raw = audio_raw[0, :]
    else:
        audio_raw, rate = librosa.load(path, sr=None, mono=True)

    return audio_raw, rate
    
def stft(audio):
    spec = librosa.stft(
        audio, n_fft=1022, hop_length=256)
    amp = np.abs(spec)
    phase = np.angle(spec)
    return torch.from_numpy(amp), torch.from_numpy(phase)
    
def main():
    audio_path = "../MUSIC_dataset/data/solo/audio/erhu/EJ-dXyKrSKA.mp3"
    heat_map_path = "heatmap.jpg"
    audio_raw , rate = load_audio_file(audio_path)
    amp, _ = stft(audio_raw)
    mag = amp.numpy()

    # Convert magnitude to a colored heatmap
    heatmap = magnitude2heatmap(mag, True)
     # Save the heatmap image (flip vertically for a spectrogram view)
    plt.imsave(heat_map_path, heatmap[::-1])
    print(f"Heatmap saved to {heat_map_path}")

if __name__ == "__main__":
    main()
