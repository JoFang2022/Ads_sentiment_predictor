import math
from typing_extensions import Self  
import pandas as pd
import numpy as np 
import glob
import os
import torch
import ffmpeg
import torch as th
from torch.utils.data import DataLoader, Dataset

class VideoLoader:
    def __init__(
            self,
            video_path,
            framerate=1,
            size=112,
            centercrop=False,
    ):
        """
        Args:
        """
        #self.video_path = video_path
        self.video_path = self.read_video(video_path)
        self.features_path = self.get_feature_path(video_path)
        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate
    
    def read_video(self,video_path): 
        if video_path.endwith(".mp4"):
            return video_path

    def get_feature_path(self, video_path):
        self.features_path = self.video_path.split(".")
        features_path = self.features_path[0] + ".npy"
        return features_path
    
    def get_video_dim(self, video_path):
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams']
                                if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        return height, width

    def get_output_dim(self, h, w):
        size = 112
        if isinstance(size, tuple) and len(size) == 2:
            return size
        elif h >= w:
            return int(h * size / w), size
        else:
            return size, int(w * size / h)

    def getitem(self, video_path):
        video_path = self.video_path
        output_file = self.features_path

        if not(os.path.isfile(output_file)) and os.path.isfile(video_path):
            print('Decoding video: {}'.format(video_path))
            try:
                h, w = self.get_video_dim(video_path)
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
        else:
            video = th.zeros(1)
            
        return {'video': video, 'input': video_path, 'output': output_file}
    
def main():
    video_path = "/Users/joliefang/Desktop/ads/New_Videos/Alwaysperiodpoverty2021.mp4"
    a = VideoLoader.getitem(video_path)
    print(a) 

if __name__ == '__main__':
    main()