import os
import glob
import argparse
import random
import fnmatch


def find_recursive(root_dir, ext='.mp3'):
    files = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, '*' + ext):
            files.append(os.path.join(root, filename))
    return files

def create_test_index(root_audio, root_frame, fps, output_dir):
    infos = []
    audio_files = find_recursive(root_audio, ext='.mp3')

    for audio_path in audio_files:
        # You might want to match your frame path convention
        frame_path = audio_path.replace(root_audio, root_frame).replace('.mp3', '.mp4')
        frame_files = glob.glob(frame_path + '/*.jpg')

        if len(frame_files) > fps * 20:
            infos.append(','.join([audio_path, frame_path, str(len(frame_files))]))

    print('{} test audio/frames pairs found.'.format(len(infos)))

    filename = os.path.join(output_dir, 'test.csv')
    with open(filename, 'w') as f:
        for item in infos:
            f.write(item + '\n')

    print('Test index saved to {}.'.format(filename))

if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_audio', default='../MUSIC_dataset/data/solo/audio',
                        help="root for extracted audio files")
    parser.add_argument('--root_frame', default='../MUSIC_dataset/data/solo/frames',
                        help="root for extracted video frames")
    parser.add_argument('--fps', default=8, type=int,
                        help="fps of video frames")
    parser.add_argument('--path_output', default='./data',
                        help="path to output index files")
    parser.add_argument('--trainset_ratio', default=0.8, type=float,
                        help="80% for training, 20% for validation")
    args = parser.parse_args()

    # find all audio/frames pairs
    infos = []
    audio_files = find_recursive(args.root_audio, ext='.mp3')
    for audio_path in audio_files:
        frame_path = audio_path.replace(args.root_audio, args.root_frame) \
                               .replace('.mp3', '.mp4')
        frame_files = glob.glob(frame_path + '/*.jpg')
        if len(frame_files) > args.fps * 20:
            infos.append(','.join([audio_path, frame_path, str(len(frame_files))]))
    print('{} audio/frames pairs found.'.format(len(infos)))

    # split train/val
    n_train = int(len(infos) * 0.8)
    random.shuffle(infos)
    trainset = infos[0:n_train]
    valset = infos[n_train:]
    for name, subset in zip(['train', 'val'], [trainset, valset]):
        filename = '{}.csv'.format(os.path.join(args.path_output, name))
        with open(filename, 'w') as f:
            for item in subset:
                f.write(item + '\n')
        print('{} items saved to {}.'.format(len(subset), filename))
'''
    root_audio = '../MUSIC_dataset/data/duet/audio'
    root_frame = '../MUSIC_dataset/data/duet/frames'
    fps = 8
    output_dir = './data'

    os.makedirs(output_dir, exist_ok=True)
    create_test_index(root_audio, root_frame, fps, output_dir)
    print('Done!')
