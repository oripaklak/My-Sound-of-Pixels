import os
import shutil

import torch
import numpy as np
import librosa
import cv2

import subprocess as sp
from threading import Timer


def warpgrid(bs, HO, WO, warp=True):
    # meshgrid
    x = np.linspace(-1, 1, WO)
    y = np.linspace(-1, 1, HO)
    xv, yv = np.meshgrid(x, y)
    grid = np.zeros((bs, HO, WO, 2))
    grid_x = xv
    if warp:
        grid_y = (np.power(21, (yv+1)/2) - 11) / 10
    else:
        grid_y = np.log(yv * 10 + 11) / np.log(21) * 2 - 1
    grid[:, :, :, 0] = grid_x
    grid[:, :, :, 1] = grid_y
    grid = grid.astype(np.float32)
    return grid


def makedirs(path, remove=False):
    if os.path.isdir(path):
        if remove:
            shutil.rmtree(path)
            print('removed existing directory...')
        else:
            return
    os.makedirs(path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val*weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        val = np.asarray(val)
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        if self.val is None:
            return 0.
        else:
            return self.val.tolist()

    def average(self):
        if self.avg is None:
            return 0.
        else:
            return self.avg.tolist()

'''
def recover_rgb(img):
    for t, m, s in zip(img,
                       [0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225]):
        t.mul_(s).add_(m)
    img = (img.numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
    return img
'''
def recover_rgb(img_tensor):
    """
    Convert a normalized tensor (C,H,W) back to uint8 RGB image (H,W,C)
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(-1, 1, 1)

    img = img_tensor * std + mean  # de-normalize
    img = img.clamp(0, 1)          # ensure values in [0,1]
    img = img.permute(1, 2, 0).cpu().numpy()  # (H,W,C)
    img = (img * 255).astype(np.uint8)
    return img

def magnitude2heatmap(mag, log=True, scale=200.):
    if log:
        mag = np.log10(mag + 1.)
    mag *= scale
    mag[mag > 255] = 255
    mag = mag.astype(np.uint8)
    mag_color = cv2.applyColorMap(mag, cv2.COLORMAP_JET)
    mag_color = mag_color[:, :, ::-1]
    return mag_color


def istft_reconstruction(mag, phase, hop_length=256):
    spec = mag.astype(complex) * np.exp(1j*phase)
    wav = librosa.istft(spec, hop_length=hop_length)
    wav = np.clip(wav, -1., 1.)
    wav = (wav * 32767).astype(np.int16)
    return wav 


class VideoWriter:
    """ Combine numpy frames into video using ffmpeg

    Arguments:
        filename: name of the output video
        fps: frame per second
        shape: shape of video frame

    Properties:
        add_frame(frame):
            add a frame to the video
        add_frames(frames):
            add multiple frames to the video
        release():
            release writing pipe

    """

    def __init__(self, filename, fps, shape):
        self.file = filename
        self.fps = fps
        self.shape = shape

        # video codec
        ext = filename.split('.')[-1]
        if ext == "mp4":
            self.vcodec = "h264"
        else:
            raise RuntimeError("Video codec not supoorted.")

        # video writing pipe
        cmd = [
            "ffmpeg",
            "-y",                                     # overwrite existing file
            "-f", "rawvideo",                         # file format
            "-s", "{}x{}".format(shape[1], shape[0]), # size of one frame
            "-pix_fmt", "rgb24",                      # 3 channels
            "-r", str(self.fps),                      # frames per second
            "-i", "-",                                # input comes from a pipe
            "-an",                                    # not to expect any audio
            "-vcodec", self.vcodec,                   # video codec
            "-pix_fmt", "yuv420p",                  # output video in yuv420p
            self.file]

        self.pipe = sp.Popen(cmd, stdin=sp.PIPE, stderr=sp.PIPE, bufsize=10**9)

    def release(self):
        self.pipe.stdin.close()
        self.pipe.wait()  # <--- Important! Wait for ffmpeg to finish.

    def add_frame(self, frame):
        assert len(frame.shape) == 3
        assert frame.shape[0] == self.shape[0]
        assert frame.shape[1] == self.shape[1]
        try:
            #self.pipe.stdin.write(frame.tostring())
            self.pipe.stdin.write(frame.tobytes())
        except:
            _, ffmpeg_error = self.pipe.communicate()
            print(ffmpeg_error)

    def add_frames(self, frames):
        for frame in frames:
            self.add_frame(frame)

class CV2VideoWriter:
    """
    Combine numpy frames into video using OpenCV instead of ffmpeg.

    Arguments:
        filename: name of the output video (e.g., 'output.mp4')
        fps: frames per second
        shape: (height, width) of video frame
    """

    def __init__(self, filename, fps, shape):
        self.filename = filename
        self.fps = fps
        self.shape = shape  # (H, W)
        
        # Use a common codec like 'mp4v' for mp4 output
        ext = filename.split('.')[-1]
        if ext == "mp4":
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif ext == "avi":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        else:
            raise RuntimeError("Unsupported video format. Use mp4 or avi.")
        
        # OpenCV expects shape as (W, H)
        self.writer = cv2.VideoWriter(filename, fourcc, fps, (shape[1], shape[0]))

    def add_frame(self, frame):
        """
        Add a single frame. Input should be RGB NumPy array (H, W, 3).
        """
        assert len(frame.shape) == 3
        assert frame.shape[0] == self.shape[0] and frame.shape[1] == self.shape[1]
        # Convert RGB to BGR because OpenCV expects BGR
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.writer.write(bgr_frame)

    def add_frames(self, frames):
        for frame in frames:
            self.add_frame(frame)

    def release(self):
        self.writer.release()

'''
def kill_proc(proc):
    proc.kill()
    print('Process running overtime! Killed.')


def run_proc_timeout(proc, timeout_sec):
    # kill_proc = lambda p: p.kill()
    timer = Timer(timeout_sec, kill_proc, [proc])
    try:
        timer.start()
        proc.communicate()
    finally:
        timer.cancel()


def combine_video_audio(src_video, src_audio, dst_video, verbose=False):
    try:
        cmd = ["ffmpeg", "-y",
               "-loglevel", "quiet",
               "-i", src_video,
               "-i", src_audio,
               "-c:v", "copy",
               "-c:a", "aac",
               "-strict", "experimental",
               dst_video]
        proc = sp.Popen(cmd)
        run_proc_timeout(proc, 10.)

        if verbose:
            print('Processed:{}'.format(dst_video))
    except Exception as e:
        print('Error:[{}] {}'.format(dst_video, e))
'''
def run_proc_timeout(proc, timeout):
    """
    Helper function to wait for the subprocess to finish with a timeout.
    """
    try:
        proc.wait(timeout=timeout)
    except sp.TimeoutExpired:
        proc.kill()
        raise RuntimeError("Process timed out")

def combine_video_audio(src_video, src_audio, dst_video, verbose=False):
    """
    Combine a video file and an audio file into a single output video.

    Arguments:
        src_video: path to the input video file (no audio or with old audio)
        src_audio: path to the input audio file
        dst_video: path to the output video file
        verbose: if True, prints the output path

    Uses ffmpeg to merge the streams.
    """
    try:
        cmd = [
            "ffmpeg",
            "-y",                     # overwrite output file if exists
            "-loglevel", "quiet",     # suppress output
            "-i", src_video,          # input video
            "-i", src_audio,          # input audio
            "-c:v", "copy",           # copy video stream as-is
            "-c:a", "aac",            # encode audio as AAC
            "-strict", "experimental",
            dst_video
        ]
        proc = sp.Popen(cmd)
        run_proc_timeout(proc, timeout=10.0)

        if verbose:
            print(f"Processed: {dst_video}")

    except Exception as e:
        print(f"Error combining video/audio [{dst_video}]: {e}")

# save video to the disk using ffmpeg
def save_video(path, tensor, fps=25):
    #print(f"path: {path}")
    assert tensor.ndim == 4, 'video should be in 4D numpy array'
    L, H, W, C = tensor.shape
    writer = CV2VideoWriter(
        path,
        fps=fps,
        shape=[H, W])
    for t in range(L):
        writer.add_frame(tensor[t])
    writer.release()


def save_audio(path, audio_numpy, sr):
    librosa.output.write_wav(path, audio_numpy, sr)