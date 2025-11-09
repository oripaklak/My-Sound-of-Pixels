import os
import random
from .base import BaseDataset

class MUSICTestDataset(BaseDataset):
    def __init__(self, list_sample, args, **kwargs):
        super(MUSICTestDataset, self).__init__(
            list_sample, args, **kwargs)
        self.fps = args.frameRate
    def __getitem__(self,index):
        path_frames = []
        path_audio, path_frame, count_frames = self.list_sample[index]

        # select frames
        center_frame = int(count_frames) // 2
        center_time = (center_frame - 0.5) / self.fps
       
        # frames path (..., center - 24, center, center + 24, ...)
        for i in range(self.num_frames):
            idx_offset = (i - self.num_frames // 2) * self.stride_frames
            path_frames.append(
                os.path.join(
                    path_frame,
                    '{:06d}.jpg'.format(center_frame + idx_offset)))

        # load frames and audios, STFT (mag + phase)
        frames = self._load_frames(path_frames)
        audio = self._load_audio(path_audio, center_time)
        mag, phase = self._stft(audio)

        # Package output
        ret_dict = {
            'mag': mag,
            'phase': phase,
            'frames': frames,
            'audio': audio,
            'info': (path_audio, path_frame, count_frames)
        }
        return ret_dict

class MUSICMixDataset(BaseDataset):
    def __init__(self, list_sample, opt, **kwargs):
        super(MUSICMixDataset, self).__init__(
            list_sample, opt, **kwargs)
        self.fps = opt.frameRate
        self.num_mix = opt.num_mix

    def __getitem__(self, index):
        N = self.num_mix
        frames = [None for n in range(N)]
        audios = [None for n in range(N)]
        infos = [[] for n in range(N)]
        path_frames = [[] for n in range(N)]
        path_audios = ['' for n in range(N)]
        center_frames = [0 for n in range(N)]

        # the first video
        infos[0] = self.list_sample[index]

        # sample other videos
        if not self.split == 'train':
            random.seed(index)
        for n in range(1, N):
            indexN = random.randint(0, len(self.list_sample)-1)
            infos[n] = self.list_sample[indexN]

        # select frames
        idx_margin = max(
            int(self.fps * 8), (self.num_frames // 2) * self.stride_frames)
        for n, infoN in enumerate(infos):
            path_audioN, path_frameN, count_framesN = infoN

            if self.split == 'train':
                # random, not to sample start and end n-frames
                center_frameN = random.randint(
                    idx_margin+1, int(count_framesN)-idx_margin)
            else:
                center_frameN = int(count_framesN) // 2
            center_frames[n] = center_frameN

            # absolute frame/audio paths
            for i in range(self.num_frames):
                idx_offset = (i - self.num_frames // 2) * self.stride_frames
                path_frames[n].append(
                    os.path.join(
                        path_frameN,
                        '{:06d}.jpg'.format(center_frameN + idx_offset)))
            path_audios[n] = path_audioN

        # load frames and audios, STFT
        try:
            for n, infoN in enumerate(infos):
                frames[n] = self._load_frames(path_frames[n])
                # jitter audio
                # center_timeN = (center_frames[n] - random.random()) / self.fps
                center_timeN = (center_frames[n] - 0.5) / self.fps
                audios[n] = self._load_audio(path_audios[n], center_timeN)
            mag_mix, mags, phase_mix = self._mix_n_and_stft(audios)

        except Exception as e:
            print('Failed loading frame/audio: {}'.format(e))
            # create dummy data
            mag_mix, mags, frames, audios, phase_mix = \
                self.dummy_mix_data(N)

        ret_dict = {'mag_mix': mag_mix, 'frames': frames, 'mags': mags}
        if self.split != 'train':
            ret_dict['audios'] = audios
            ret_dict['phase_mix'] = phase_mix
            ret_dict['infos'] = infos
        ret_dict['center_frames'] = center_frames

        return ret_dict
    
        '''
        # Collect paths for ALL frames
        path_frames = []
        for frame_idx in range(1, int(count_framesN) + 1):
            frame_path = os.path.join(
                path_frameN,
                '{:06d}.jpg'.format(frame_idx)
            )
            path_frames.append(frame_path)

        # Load ALL frames
        frames = self._load_frames(path_frames)

        # Load audio for the whole clip 
        audio  = self._load_audio_whole(path_audioN)

        # Compute STFT for the full audio
        mag, phase = self._stft(audio)
        '''
