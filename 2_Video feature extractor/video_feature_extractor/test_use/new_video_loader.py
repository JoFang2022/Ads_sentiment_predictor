import torch as th
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import ffmpeg
from Read_Video import *


class VideoLoader(Dataset):
    """Pytorch video loader."""
    
    # def read_video(self, Dataset):
    #     if Dataset.endswith(".mp4"):
    #         video_path = Dataset
    #         features_path = Dataset.split(".")
    #         features_path = features_path[0] + ".npy"
    #         return features_path, video_path
    
    def __init__(
            self,
            video_path,
            features_path,
            framerate=1,
            size=112,
            centercrop=False,
    ):
        """
        Args:
        """
        self.video_path = read_video(video_path)
        self.features_path = get_feature_path(video_path)
        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate

    # def __len__(self):
    #     return len(self.video_path)

    def _get_video_dim(self, video_path):
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams']
                             if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        return height, width

    def _get_output_dim(self, h, w):
        if isinstance(self.size, tuple) and len(self.size) == 2:
            return self.size
        elif h >= w:
            return int(h * self.size / w), self.size
        else:
            return self.size, int(w * self.size / h)

    def __getitem__(self,video_path):
        video_path = self.video_path
        output_file = self.features_path

        # if not(os.path.isfile(output_file)) and os.path.isfile(video_path):
        #     print('Decoding video: {}'.format(video_path))
        try:
            h, w = self._get_video_dim(video_path)
        except:
            print('ffprobe failed at: {}'.format(video_path))
            return {'video': th.zeros(1), 'input': video_path,
                    'output': output_file}
        height, width = self._get_output_dim(h, w)
        cmd = (
            ffmpeg
            .input(video_path)
            .filter('fps', fps=self.framerate)
            .filter('scale', width, height)
        )
        if self.centercrop:
            x = int((width - self.size) / 2.0)
            y = int((height - self.size) / 2.0)
            cmd = cmd.crop(x, y, self.size, self.size)
        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, quiet=True)
        )
        if self.centercrop and isinstance(self.size, int):
            height, width = self.size, self.size
        video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
        video = th.from_numpy(video.astype('float32'))
        video = video.permute(0, 3, 1, 2)

        return {'video': video,'input': video_path, 'output': output_file}
