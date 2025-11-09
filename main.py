# System libs
import json
import os
import random
import time

# Numerical libs
import torch
import torch.nn.functional as F
import numpy as np
import scipy.io.wavfile as wavfile
import imageio.v2 as imageio # !!added!!
#from scipy.misc import imsave
from mir_eval.separation import bss_eval_sources

# Training libs
import wandb

# Our libs
from arguments import ArgParser
from dataset import MUSICMixDataset, MUSICTestDataset
from models import ModelBuilder, ModelTest, activate
from utils import AverageMeter, \
    recover_rgb, magnitude2heatmap,\
    istft_reconstruction, warpgrid, \
    combine_video_audio, save_video, makedirs
from viz import plot_loss_metrics, HTMLVisualizer

# Network wrapper, defines forward pass
class NetWrapper(torch.nn.Module):
    def __init__(self, nets, crit):
        super(NetWrapper, self).__init__()
        self.net_sound, self.net_frame, self.net_synthesizer = nets
        self.crit = crit

    def forward(self, batch_data, args):
        mag_mix = batch_data['mag_mix']
        device = mag_mix.device if mag_mix.is_cuda else args.device
        if not mag_mix.is_cuda:
            mag_mix = mag_mix.to(device, non_blocking=True)
        mags = [mag if mag.is_cuda else mag.to(device, non_blocking=True)
                for mag in batch_data['mags']]
        frames = [frame if frame.is_cuda else frame.to(device, non_blocking=True)
                  for frame in batch_data['frames']]
        #mag_mix = torch.tensor(batch_data['mag_mix']).to(args.device)  # shape: (B, 1, HS, WS)
        #mags =  torch.tensor(batch_data['mags']).to(args.device)  # shape: (B, N, 1, HS, WS)
        #frames =  torch.tensor(batch_data['frames']).to(args.device)  # shape: (B, C, T, HI, WI)
        mag_mix = mag_mix + 1e-10

        N = args.num_mix
        B = mag_mix.size(0)
        T = mag_mix.size(3)

        # 0.0 warp the spectrogram
        if args.log_freq:
            grid_warp = torch.from_numpy(
                warpgrid(B, 256, T, warp=True)).to(device)
            mag_mix = F.grid_sample(mag_mix, grid_warp)
            for n in range(N):
                mags[n] = F.grid_sample(mags[n], grid_warp)

        # 0.1 calculate loss weighting coefficient: magnitude of input mixture
        if args.weighted_loss:
            weight = torch.log1p(mag_mix)
            weight = torch.clamp(weight, 1e-3, 10)
        else:
            weight = torch.ones_like(mag_mix)

        # 0.2 ground truth masks are computed after warpping!
        gt_masks = [None for n in range(N)]
        for n in range(N):
            if args.binary_mask:
                # for simplicity, mag_N > 0.5 * mag_mix
                gt_masks[n] = (mags[n] > 0.5 * mag_mix).float()
            else:
                gt_masks[n] = mags[n] / mag_mix
                # clamp to avoid large numbers in ratio masks
                gt_masks[n].clamp_(0., 5.)

        # LOG magnitude
        log_mag_mix = torch.log(mag_mix).detach()

        # 1. forward net_sound -> BxCxHxW
        feat_sound = self.net_sound(log_mag_mix)
        feat_sound = activate(feat_sound, args.sound_activation)

        # 2. forward net_frame -> Bx1xC
        feat_frames = [None for n in range(N)]
        for n in range(N):
            feat_frames[n] = self.net_frame.forward_multiframe(frames[n])
            feat_frames[n] = activate(feat_frames[n], args.img_activation)

        # 3. sound synthesizer
        pred_masks = [None for n in range(N)]
        for n in range(N):
            pred_masks[n] = self.net_synthesizer(feat_frames[n], feat_sound)
            pred_masks[n] = activate(pred_masks[n], args.output_activation)

        # 4. loss
        err = self.crit(pred_masks, gt_masks, weight).reshape(1)

        return err, \
            {'pred_masks': pred_masks, 'gt_masks': gt_masks,
             'mag_mix': mag_mix, 'mags': mags, 'weight': weight}


# Calculate metrics
def calc_metrics(batch_data, outputs, args):
    # meters
    sdr_mix_meter = AverageMeter()
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()

    # fetch data and predictions
    mag_mix = batch_data['mag_mix']
    phase_mix = batch_data['phase_mix']
    audios = batch_data['audios']

    pred_masks_ = outputs['pred_masks']

    # unwarp log scale
    N = args.num_mix
    B = mag_mix.size(0)
    pred_masks_linear = [None for n in range(N)]
    for n in range(N):
        if args.log_freq:
            grid_unwarp = torch.from_numpy(
                warpgrid(B, args.stft_frame//2+1, pred_masks_[0].size(3), warp=False)).to(args.device)
            pred_masks_linear[n] = F.grid_sample(pred_masks_[n], grid_unwarp)
        else:
            pred_masks_linear[n] = pred_masks_[n]

    # convert into numpy
    mag_mix = mag_mix.numpy()
    phase_mix = phase_mix.numpy()
    for n in range(N):
        pred_masks_linear[n] = pred_masks_linear[n].detach().cpu().numpy()

        # threshold if binary mask
        if args.binary_mask:
            pred_masks_linear[n] = (pred_masks_linear[n] > args.mask_thres).astype(np.float32)

    # loop over each sample
    for j in range(B):
        # save mixture
        mix_wav = istft_reconstruction(mag_mix[j, 0], phase_mix[j, 0], hop_length=args.stft_hop)

        # save each component
        preds_wav = [None for n in range(N)]
        for n in range(N):
            # Predicted audio recovery
            pred_mag = mag_mix[j, 0] * pred_masks_linear[n][j, 0]
            preds_wav[n] = istft_reconstruction(pred_mag, phase_mix[j, 0], hop_length=args.stft_hop)

        # separation performance computes
        L = preds_wav[0].shape[0]
        gts_wav = [None for n in range(N)]
        valid = True
        for n in range(N):
            gts_wav[n] = audios[n][j, 0:L].numpy()
            valid *= np.sum(np.abs(gts_wav[n])) > 1e-5
            valid *= np.sum(np.abs(preds_wav[n])) > 1e-5
        if valid:
            sdr, sir, sar, _ = bss_eval_sources(
                np.asarray(gts_wav),
                np.asarray(preds_wav),
                False)
            sdr_mix, _, _, _ = bss_eval_sources(
                np.asarray(gts_wav),
                np.asarray([mix_wav[0:L] for n in range(N)]),
                False)
            sdr_mix_meter.update(sdr_mix.mean())
            sdr_meter.update(sdr.mean())
            sir_meter.update(sir.mean())
            sar_meter.update(sar.mean())

    return [sdr_mix_meter.average(),
            sdr_meter.average(),
            sir_meter.average(),
            sar_meter.average()]


# Visualize predictions
def output_visuals(vis_rows, batch_data, outputs, args):
    # fetch data and predictions
    mag_mix = batch_data['mag_mix']
    phase_mix = batch_data['phase_mix']
    frames = batch_data['frames']
    infos = batch_data['infos']

    pred_masks_ = outputs['pred_masks']
    gt_masks_ = outputs['gt_masks']
    mag_mix_ = outputs['mag_mix']
    weight_ = outputs['weight']

    # unwarp log scale
    N = args.num_mix
    B = mag_mix.size(0)
    pred_masks_linear = [None for n in range(N)]
    gt_masks_linear = [None for n in range(N)]
    for n in range(N):
        if args.log_freq:
            grid_unwarp = torch.from_numpy(
                warpgrid(B, args.stft_frame//2+1, gt_masks_[0].size(3), warp=False)).to(args.device)
            pred_masks_linear[n] = F.grid_sample(pred_masks_[n], grid_unwarp)
            gt_masks_linear[n] = F.grid_sample(gt_masks_[n], grid_unwarp)
        else:
            pred_masks_linear[n] = pred_masks_[n]
            gt_masks_linear[n] = gt_masks_[n]

    # convert into numpy
    mag_mix = mag_mix.numpy()
    mag_mix_ = mag_mix_.detach().cpu().numpy()
    phase_mix = phase_mix.numpy()
    weight_ = weight_.detach().cpu().numpy()
    for n in range(N):
        pred_masks_[n] = pred_masks_[n].detach().cpu().numpy()
        pred_masks_linear[n] = pred_masks_linear[n].detach().cpu().numpy()
        gt_masks_[n] = gt_masks_[n].detach().cpu().numpy()
        gt_masks_linear[n] = gt_masks_linear[n].detach().cpu().numpy()

        # threshold if binary mask
        if args.binary_mask:
            pred_masks_[n] = (pred_masks_[n] > args.mask_thres).astype(np.float32)
            pred_masks_linear[n] = (pred_masks_linear[n] > args.mask_thres).astype(np.float32)

    center_frames_np = None
    if 'center_frames' in batch_data:
        centers = batch_data['center_frames']
        if isinstance(centers, torch.Tensor):
            centers = centers.detach().cpu().numpy()
        else:
            centers = np.asarray(centers)
        if centers.ndim == 1:
            if B == 1:
                centers = centers.reshape(1, -1)
            else:
                centers = np.expand_dims(centers, axis=1)
        if centers.ndim == 2 and centers.shape[0] != B and centers.shape[1] == B:
            centers = np.transpose(centers, (1, 0))
        center_frames_np = centers
    aud_duration = float(args.audLen) / float(args.audRate)

    # loop over each sample
    for j in range(B):
        row_elements = []
        source_meta = []

        # video names
        prefix = []
        #args.vis = os.path.join(args.vis, 'eval/')
        for n in range(N):
            prefix.append('-'.join(infos[n][0][j].split('/')[-2:]).split('.')[0])
        prefix = '+'.join(prefix)
        makedirs(os.path.join(args.vis, prefix))

        # save mixture
        mix_wav = istft_reconstruction(mag_mix[j, 0], phase_mix[j, 0], hop_length=args.stft_hop)
        mix_amp = magnitude2heatmap(mag_mix_[j, 0])
        weight = magnitude2heatmap(weight_[j, 0], log=False, scale=100.)
        filename_mixwav = os.path.join(prefix, 'mix.wav')
        filename_mixmag = os.path.join(prefix, 'mix.jpg')
        filename_weight = os.path.join(prefix, 'weight.jpg')
        imageio.imwrite(os.path.join(args.vis, filename_mixmag), mix_amp[::-1, :, :])
        imageio.imwrite(os.path.join(args.vis, filename_weight), weight[::-1, :])
        wavfile.write(os.path.join(args.vis, filename_mixwav), args.audRate, mix_wav)
        row_elements += [{'text': prefix}, {'image': filename_mixmag, 'audio': filename_mixwav}]

        # save each component
        preds_wav = [None for n in range(N)]
        for n in range(N):
            # GT and predicted audio recovery
            gt_mag = mag_mix[j, 0] * gt_masks_linear[n][j, 0]
            gt_wav = istft_reconstruction(gt_mag, phase_mix[j, 0], hop_length=args.stft_hop)
            pred_mag = mag_mix[j, 0] * pred_masks_linear[n][j, 0]
            preds_wav[n] = istft_reconstruction(pred_mag, phase_mix[j, 0], hop_length=args.stft_hop)

            # output masks
            filename_gtmask = os.path.join(prefix, 'gtmask{}.jpg'.format(n+1))
            filename_predmask = os.path.join(prefix, 'predmask{}.jpg'.format(n+1))
            gt_mask = (np.clip(gt_masks_[n][j, 0], 0, 1) * 255).astype(np.uint8)
            pred_mask = (np.clip(pred_masks_[n][j, 0], 0, 1) * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(args.vis, filename_gtmask), gt_mask[::-1, :])
            imageio.imwrite(os.path.join(args.vis, filename_predmask), pred_mask[::-1, :])

            # ouput spectrogram (log of magnitude, show colormap)
            filename_gtmag = os.path.join(prefix, 'gtamp{}.jpg'.format(n+1))
            filename_predmag = os.path.join(prefix, 'predamp{}.jpg'.format(n+1))
            gt_mag = magnitude2heatmap(gt_mag)
            pred_mag = magnitude2heatmap(pred_mag)
            imageio.imwrite(os.path.join(args.vis, filename_gtmag), gt_mag[::-1, :, :])
            imageio.imwrite(os.path.join(args.vis, filename_predmag), pred_mag[::-1, :, :])
            
            # output audio
            filename_gtwav = os.path.join(prefix, 'gt{}.wav'.format(n+1))
            filename_predwav = os.path.join(prefix, 'pred{}.wav'.format(n+1))
            wavfile.write(os.path.join(args.vis, filename_gtwav), args.audRate, gt_wav)
            wavfile.write(os.path.join(args.vis, filename_predwav), args.audRate, preds_wav[n])

            # output video
            frames_tensor = [recover_rgb(frames[n][j][t]) for t in range(args.num_frames)]
            frames_tensor = np.asarray(frames_tensor)
            path_video = os.path.join(args.vis, prefix, 'video{}.mp4'.format(n+1))
            save_video(path_video, frames_tensor, fps=args.frameRate/args.stride_frames)

            # combine gt video and audio
            filename_av = os.path.join(prefix, 'av{}.mp4'.format(n+1))
            combine_video_audio(
                path_video,
                os.path.join(args.vis, filename_gtwav),
                os.path.join(args.vis, filename_av))

            video_info = infos[n][0][j]
            youtube_id = os.path.splitext(os.path.basename(video_info))[0]
            count_entry = infos[n][2][j] if len(infos[n]) > 2 else 0
            try:
                count_frames = int(float(count_entry))
            except (TypeError, ValueError):
                count_frames = 0

            center_frame = None
            if center_frames_np is not None:
                try:
                    center_val = center_frames_np[j, n]
                except IndexError:
                    try:
                        center_val = center_frames_np[n, j]
                    except IndexError:
                        center_val = None
                if center_val is not None:
                    center_frame = int(center_val)
            if (center_frame is None or center_frame < 0) and count_frames > 0:
                center_frame = count_frames // 2
            if center_frame is None:
                center_frame = 0

            video_duration = (count_frames / args.frameRate) if (count_frames and args.frameRate) else aud_duration
            center_time = ((center_frame - 0.5) / args.frameRate) if args.frameRate else 0.0
            center_time = max(0.0, center_time)
            start_time = max(0.0, center_time - aud_duration / 2.0)
            end_time = start_time + aud_duration
            if video_duration:
                end_time = min(video_duration, end_time)
            if end_time < start_time:
                end_time = start_time

            youtube_meta = {
                'id': youtube_id,
                'start': float(round(start_time, 2)),
                'end': float(round(end_time, 2)),
                'center_frame': int(center_frame),
                'video_duration': float(round(video_duration, 2))
            }
            source_meta.append(youtube_meta)

            video_cell = {}
            if youtube_id:
                video_cell['youtube'] = youtube_meta
            elif os.path.exists(os.path.join(args.vis, filename_av)):
                video_cell['video'] = filename_av
            else:
                video_cell['text'] = '—'

            row_elements += [
                video_cell,
                {'image': filename_predmag, 'audio': filename_predwav},
                {'image': filename_gtmag, 'audio': filename_gtwav},
                {'image': filename_predmask},
                {'image': filename_gtmask}]

        row_elements += [{'image': filename_weight}]

        meta_payload = {
            'sources': source_meta,
            'audio_duration': float(round(aud_duration, 2)),
            'frame_rate': float(args.frameRate),
            'sample_rate': int(args.audRate)
        }
        meta_path = os.path.join(args.vis, prefix, 'meta.json')
        try:
            with open(meta_path, 'w') as meta_file:
                json.dump(meta_payload, meta_file, indent=2)
        except IOError as io_err:
            print(f'[Meta] Failed to write metadata for {prefix}: {io_err}')

        vis_rows.append(row_elements)

def output_visuals_test(batch_data, outputs, args):
    """
    Writes per-pixel predictions for the test split without keeping all CUDA
    tensors resident. Each pixel is processed sequentially to keep the memory
    footprint small while still generating the same artifacts as before.
    """
    outputs = outputs.detach()
    if outputs.is_cuda:
        outputs = outputs.cpu()

    B, HI, WI, HS, WS = outputs.size()
    mag = batch_data["mag"].detach().cpu().numpy()        # (B, 1, F, T)
    phase = batch_data["phase"].detach().cpu().numpy()    # (B, 1, F, T)
    infos = batch_data["info"]

    grid_unwarp = None
    if args.log_freq:
        grid_unwarp = torch.from_numpy(
            warpgrid(1, args.stft_frame // 2 + 1, WS, warp=False)
        ).type_as(outputs)

    for j in range(B):
        prefix = []
        prefix.append('-'.join(infos[0][0].split('/')[-2:]).split('.')[0])
        prefix = '+'.join(prefix)

        root_dir = os.path.join(args.vis, 'test', prefix)
        pred_mask_dir = os.path.join(root_dir, 'pred masks')
        pred_audio_dir = os.path.join(root_dir, 'pred audio')
        pred_heatmap_dir = os.path.join(root_dir, 'pred heat map')

        makedirs(root_dir)
        makedirs(pred_mask_dir)
        makedirs(pred_audio_dir)
        makedirs(pred_heatmap_dir)

        mag_sample = mag[j, 0]
        phase_sample = phase[j, 0]
        time_bins = mag_sample.shape[-1]
        waveform_time = max(1, (time_bins - 1) * args.stft_hop)

        result = np.zeros((waveform_time, HI, WI), dtype=np.float32)
        
        for h in range(HI):
            for w in range(WI):
                pred = outputs[j, h, w].unsqueeze(0).unsqueeze(0)  # (1,1,HS,WS)
                if args.log_freq:
                    pred = F.grid_sample(pred, grid_unwarp, align_corners=False)

                pred_np = pred.squeeze().numpy()
                if args.binary_mask:
                    pred_np = (pred_np > args.mask_thres).astype(np.float32)

                pred_mag = mag_sample * pred_np
                pred_wav = istft_reconstruction(pred_mag, phase_sample, hop_length=args.stft_hop)

                result[:, h, w] = pred_wav

                filename_predmask = f'predmask{h+1}x{w+1}.jpg'
                pred_mask_img = (np.clip(pred_np, 0, 1) * 255).astype(np.uint8)
                imageio.imwrite(os.path.join(pred_mask_dir, filename_predmask), pred_mask_img[::-1, :])

                filename_predmag = f'predamp{h+1}x{w+1}.jpg'
                pred_heatmap = magnitude2heatmap(pred_mag)
                imageio.imwrite(os.path.join(pred_heatmap_dir, filename_predmag), pred_heatmap[::-1, :, :])

                filename_predwav = f'pred{h+1}x{w+1}.wav'
                wavfile.write(os.path.join(pred_audio_dir, filename_predwav), args.audRate, pred_wav)
        print(result[100:103, :, :])
        activity_dir = os.path.join(root_dir, 'pixel activity')
        makedirs(activity_dir)
        activity_video = os.path.join(activity_dir, 'pixel_activity.mp4')
        abs_threshold = getattr(args, 'pixel_activity_abs_threshold', 1e-3)
        rel_threshold = getattr(args, 'pixel_activity_rel_threshold', 0.2)
        samples_per_frame = getattr(args, 'pixel_activity_samples_per_frame', args.stft_hop * 4)
        energy_gamma = getattr(args, 'pixel_activity_energy_gamma', 2.0)
        rel_threshold = float(np.clip(rel_threshold, 0.0, 1.0))
        create_noise_activity_video(
            result,
            activity_video,
            sample_rate=args.audRate,
            abs_threshold=abs_threshold,
            rel_threshold=rel_threshold,
            samples_per_frame=max(1, int(samples_per_frame)),
            energy_gamma=max(1.0, float(energy_gamma)),
            active_color=(255, 0, 0),
            inactive_color=(255, 255, 255))

def create_noise_activity_video(waveform_cube,
                                output_path,
                                sample_rate,
                                abs_threshold=1e-3,
                                rel_threshold=0.3,
                                samples_per_frame=1024,
                                energy_gamma=2.0,
                                active_color=(255, 0, 0),
                                inactive_color=(255, 255, 255),
                                eps=1e-8):
    """
    Convert a (T, HI, WI) waveform tensor into a color-coded activity video.

    Args:
        waveform_cube (np.ndarray): Array with shape (time_samples, HI, WI) containing
            reconstructed waveforms for each pixel.
        output_path (str): Destination video filename (.mp4 or .avi supported).
        sample_rate (float): Waveform sampling rate (Hz).
        abs_threshold (float): Minimum RMS value to consider a pixel active.
        rel_threshold (float): Fraction within [0, 1]; pixels whose RMS exceeds
            `median + rel_threshold * (max - median)` are counted as active.
        samples_per_frame (int): Number of raw samples aggregated into a single frame.
        energy_gamma (float): Exponent applied to per-pixel RMS values to stretch their
            dynamic range before thresholding (gamma >= 1).
            FPS will be derived as sample_rate / samples_per_frame by the caller.
        active_color (Tuple[int, int, int]): RGB color for active pixels.
        inactive_color (Tuple[int, int, int]): RGB color for silent pixels.
        eps (float): Numerical stability constant for RMS computation.
    """
    waveform_np = np.asarray(waveform_cube, dtype=np.float32)
    if waveform_np.ndim != 3:
        raise ValueError(f"waveform_cube must be 3D (T, HI, WI); got {waveform_np.shape}")

    T, HI, WI = waveform_np.shape
    samples_per_frame = max(1, int(samples_per_frame))
    num_frames = max(1, int(np.ceil(T / samples_per_frame)))

    active_rgb = np.asarray(active_color, dtype=np.uint8)
    inactive_rgb = np.asarray(inactive_color, dtype=np.uint8)
    frames = np.empty((num_frames, HI, WI, 3), dtype=np.uint8)

    for idx in range(num_frames):
        start = idx * samples_per_frame
        end = min((idx + 1) * samples_per_frame, T)
        chunk = waveform_np[start:end]
        rms = np.sqrt(np.mean(chunk * chunk, axis=0) + eps)
        if energy_gamma is not None and energy_gamma > 1:
            rms = np.power(rms, energy_gamma)
        frame_max = float(rms.max())
        frame_median = float(np.median(rms))
        dynamic = frame_max - frame_median
        adaptive = frame_median + float(rel_threshold) * dynamic if dynamic > 0 else frame_max
        cutoff = max(abs_threshold, adaptive)
        activity = rms >= cutoff if frame_max > 0 else np.zeros_like(rms, dtype=bool)
        frame = np.where(activity[..., None], active_rgb, inactive_rgb)
        frames[idx] = frame.astype(np.uint8)

    fps = max(sample_rate / max(samples_per_frame, 1), 1e-3)
    save_video(output_path, frames, fps=max(fps, 1e-3))
    return frames


def evaluate(netWrapper, loader, history, epoch, args):
    print('Evaluating at {} epochs...'.format(epoch))
    torch.set_grad_enabled(False)
    
    # remove previous viz results
    makedirs(args.vis, remove=True)

    # switch to eval mode
    netWrapper.eval()

    # initialize meters
    loss_meter = AverageMeter()
    sdr_mix_meter = AverageMeter()
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()
    
    # initialize HTML header
    visualizer = HTMLVisualizer(os.path.join(args.vis, 'index.html'))
    header = ['Filename', 'Input Mixed Audio']
    for n in range(1, args.num_mix+1):
        header += ['Video {:d}'.format(n),
                   'Predicted Audio {:d}'.format(n),
                   'GroundTruth Audio {}'.format(n),
                   'Predicted Mask {}'.format(n),
                   'GroundTruth Mask {}'.format(n)]
    header += ['Loss weighting']
    visualizer.add_header(header)
    
    vis_rows = []

    for i, batch_data in enumerate(loader):
        # forward pass
        err, outputs = netWrapper.forward(batch_data, args)        
        err = err.mean()

        loss_meter.update(err.item())
        print('[Eval] iter {}, loss: {:.4f}'.format(i, err.item()))

        # calculate metrics
        sdr_mix, sdr, sir, sar = calc_metrics(batch_data, outputs, args)
        sdr_mix_meter.update(sdr_mix)
        sdr_meter.update(sdr)
        sir_meter.update(sir)
        sar_meter.update(sar)

        # output visualization
        if len(vis_rows) < args.num_vis:
            output_visuals(vis_rows, batch_data, outputs, args)

    print('[Eval Summary] Epoch: {}, Loss: {:.4f}, '
          'SDR_mixture: {:.4f}, SDR: {:.4f}, SIR: {:.4f}, SAR: {:.4f}'
          .format(epoch, loss_meter.average(),
                  sdr_mix_meter.average(),
                  sdr_meter.average(),
                  sir_meter.average(),
                  sar_meter.average()))
    
    wandb.log({
        "val/loss": loss_meter.average(),
        "val/sdr_mix": sdr_mix_meter.average(),
        "val/sdr": sdr_meter.average(),
        "val/sir": sir_meter.average(),
        "val/sar": sar_meter.average(),
        "val/epoch": epoch
    })

    history['val']['epoch'].append(epoch)
    history['val']['err'].append(loss_meter.average())
    history['val']['sdr'].append(sdr_meter.average())
    history['val']['sir'].append(sir_meter.average())
    history['val']['sar'].append(sar_meter.average())

    print('Plotting html for visualization...')
    visualizer.add_rows(vis_rows)
    visualizer.write_html()

    # Plot figure
    if epoch > 0:
        print('Plotting figures...')
        plot_loss_metrics(args.ckpt, history)


# train one epoch
# use wandb
def train(netWrapper, loader, optimizer, history, epoch, args):
    torch.set_grad_enabled(True)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # switch to train mode
    netWrapper.train()

    # main loop
    if args.device.type == 'cuda':
        torch.cuda.synchronize()
    tic = time.perf_counter()
    for i, batch_data in enumerate(loader):
        # measure data time
        if args.device.type == 'cuda':
            torch.cuda.synchronize()
        data_time.update(time.perf_counter() - tic)
        
        # forward pass
        netWrapper.zero_grad()
        err, _ = netWrapper.forward(batch_data, args)
        err = err.mean()

        # backward
        err.backward()
        optimizer.step()

        # measure total time
        if args.device.type == 'cuda':
            torch.cuda.synchronize()
        batch_time.update(time.perf_counter() - tic)
        tic = time.perf_counter()

        # display
        if i % args.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_sound: {}, lr_frame: {}, lr_synthesizer: {}, '
                  'loss: {:.4f}'
                  .format(epoch, i, args.epoch_iters,
                          batch_time.average(), data_time.average(),
                          args.lr_sound, args.lr_frame, args.lr_synthesizer,
                          err.item()))
            fractional_epoch = epoch - 1 + 1. * i / args.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['err'].append(err.item())

            wandb.log({
                "train/loss": err.item(),
                "train/epoch": fractional_epoch,
                "train/batch_time": batch_time.average(),
                "train/data_time": data_time.average(),
                "lr_sound": args.lr_sound,
                "lr_frame": args.lr_frame,
                "lr_synthesizer": args.lr_synthesizer
            })

def test(nets, loader, args):
    ModelTest_ = ModelTest(nets)
    if args.num_gpus > 1:
        ModelTest_ = torch.nn.DataParallel(ModelTest_, device_ids=range(args.num_gpus))
    ModelTest_.to(args.device)
    for i, batch_data in enumerate(loader):
        print('[Test] iter {}'.format(i))
        pred_masks = ModelTest_(batch_data ,args)
        
        output_visuals_test(batch_data, pred_masks, args)
  
def checkpoint(nets, history, epoch, args):
    print('Saving checkpoints at {} epochs.'.format(epoch))
    (net_sound, net_frame, net_synthesizer) = nets
    suffix_latest = 'latest.pth'
    suffix_best = 'best.pth'
    '''
    torch.save(history,
               '{}/history_{}'.format(args.ckpt, suffix_latest))
    torch.save(net_sound.state_dict(),
               '{}/sound_{}'.format(args.ckpt, suffix_latest))
    torch.save(net_frame.state_dict(),
               '{}/frame_{}'.format(args.ckpt, suffix_latest))
    torch.save(net_synthesizer.state_dict(),
               '{}/synthesizer_{}'.format(args.ckpt, suffix_latest))
    '''
    cur_err = history['val']['err'][-1]
    if cur_err < args.best_err:
        args.best_err = cur_err
        torch.save(net_sound.state_dict(),
                   '{}/sound_{}'.format(args.ckpt, suffix_best))
        torch.save(net_frame.state_dict(),
                   '{}/frame_{}'.format(args.ckpt, suffix_best))
        torch.save(net_synthesizer.state_dict(),
                   '{}/synthesizer_{}'.format(args.ckpt, suffix_best))


def create_optimizer(nets, args):
    (net_sound, net_frame, net_synthesizer) = nets
    param_groups = [{'params': net_sound.parameters(), 'lr': args.lr_sound},
                    {'params': net_synthesizer.parameters(), 'lr': args.lr_synthesizer},
                    {'params': net_frame.features.parameters(), 'lr': args.lr_frame},
                    {'params': net_frame.fc.parameters(), 'lr': args.lr_sound}]
    return torch.optim.SGD(param_groups, momentum=args.beta1, weight_decay=args.weight_decay)


def adjust_learning_rate(optimizer, args):
    args.lr_sound *= 0.1
    args.lr_frame *= 0.1
    args.lr_synthesizer *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1


def main(args):
    # Network Builders
    builder = ModelBuilder()
    net_sound = builder.build_sound(
        arch=args.arch_sound,
        fc_dim=args.num_channels,
        weights=args.weights_sound)
    net_frame = builder.build_frame(
        arch=args.arch_frame,
        fc_dim=args.num_channels,
        pool_type=args.img_pool,
        weights=args.weights_frame)
    net_synthesizer = builder.build_synthesizer(
        arch=args.arch_synthesizer,
        fc_dim=args.num_channels,
        weights=args.weights_synthesizer)
    nets = (net_sound, net_frame, net_synthesizer)
    crit = builder.build_criterion(arch=args.loss)

    # Dataset and Loader
    dataset_train = MUSICMixDataset(
        args.list_train, args, split='train')
    dataset_val = MUSICMixDataset(
        args.list_val, args, max_sample=args.num_val, split='val')
    dataset_test = MUSICTestDataset(
        args.list_test, args, split='test')
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        drop_last=True,
        pin_memory=args.device.type == 'cuda',
        persistent_workers=int(args.workers) > 0)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False,
        pin_memory=args.device.type == 'cuda',
        persistent_workers=True)
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False,
        pin_memory=args.device.type == 'cuda',
        persistent_workers=True)
    # Test mode
    if args.mode == 'test':
        test(nets, loader_test, args)
        print('Test Done!')
        return
    
    # Eval or Train mode 
    args.epoch_iters = len(dataset_train) // args.batch_size
    print('1 Epoch = {} iters'.format(args.epoch_iters))
    
    
    # Wrap networks
    netWrapper = NetWrapper(nets, crit)
    netWrapper = torch.nn.DataParallel(netWrapper, device_ids=gpu_id)
    #netWrapper = torch.nn.DataParallel(netWrapper, device_ids=range(args.num_gpus))
    netWrapper.to(args.device)
    
    # Set up optimizer
    optimizer = create_optimizer(nets, args)

    # History of peroformance
    history = {
        'train': {'epoch': [], 'err': []},
        'val': {'epoch': [], 'err': [], 'sdr': [], 'sir': [], 'sar': []}}
    
    # ✅ Initialize Weights & Biases
    wandb.init(
        project="my-sound-of-pixels",
        entity="oripaklak-bar-ilan-university",   
        config=vars(args)     # log all hyperparameters from args
    )

    # Eval mode
    evaluate(netWrapper, loader_val, history, 0, args)
    if args.mode == 'eval':
        print('Evaluation Done!')
        return

    # Training loop
    for epoch in range(1, args.num_epoch + 1):
        train(netWrapper, loader_train, optimizer, history, epoch, args)

        # Evaluation and visualization
        if epoch % args.eval_epoch == 0:
            evaluate(netWrapper, loader_val, history, epoch, args)

            # checkpointing
            checkpoint(nets, history, epoch, args)

        # drop learning rate
        if epoch in args.lr_steps:
            adjust_learning_rate(optimizer, args)

    wandb.finish()  # finish the wandb run
    print('Training Done!')


if __name__ == '__main__':
    # arguments
    parser = ArgParser()
    args = parser.parse_train_arguments()
    requested_gpus = args.num_gpus
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        if requested_gpus > available_gpus:
            print('Requested {} GPUs but only {} available. Using {} instead.'.format(
                requested_gpus, available_gpus, available_gpus))
            args.num_gpus = available_gpus
    else:
        raise RuntimeError('CUDA is not available. Please run on a machine with at least one GPU.')

    if args.num_gpus == 0:
        raise RuntimeError('No CUDA devices visible. Please check your CUDA configuration or lower --num_gpus.')

    gpu_id = list(range(args.num_gpus))
    args.device = torch.device('cuda', gpu_id[0])
    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    # experiment name
    if args.mode == 'train':
        args.id += '-{}mix'.format(args.num_mix)
        if args.log_freq:
            args.id += '-LogFreq'
        args.id += '-{}-{}-{}'.format(
            args.arch_frame, args.arch_sound, args.arch_synthesizer)
        args.id += '-frames{}stride{}'.format(args.num_frames, args.stride_frames)
        args.id += '-{}'.format(args.img_pool)
        if args.binary_mask:
            assert args.loss == 'bce', 'Binary Mask should go with BCE loss'
            args.id += '-binary'
        else:
            args.id += '-ratio'
        if args.weighted_loss:
            args.id += '-weightedLoss'
        args.id += '-channels{}'.format(args.num_channels)
        args.id += '-epoch{}'.format(args.num_epoch)
        args.id += '-step' + '_'.join([str(x) for x in args.lr_steps])

    print('Model ID: {}'.format(args.id))

    # paths to save/load output
    args.ckpt = os.path.join(args.ckpt, args.id)
    args.vis = os.path.join(args.ckpt, 'visualization/')
    if args.mode == 'test':
        args.vis = os.path.join(args.ckpt, 'test/')
    if args.mode == 'train':
        makedirs(args.ckpt, remove=True)
    elif args.mode == 'eval' or args.mode == 'test':
        args.weights_sound = os.path.join(args.ckpt, 'sound_best.pth')
        args.weights_frame = os.path.join(args.ckpt ,'frame_best.pth')
        args.weights_synthesizer = os.path.join(args.ckpt, 'synthesizer_best.pth')

    # initialize best error with a big number
    args.best_err = float("inf")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
