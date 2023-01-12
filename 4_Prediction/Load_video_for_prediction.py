import decord
from decord import VideoReader
import torch as th


class Video_Reader:
    def __init__(self):
        self = self

    def read_video(self, video_path):
        decord.bridge.set_bridge("torch")
        vr_tensor = VideoReader(video_path)
        torch_tensor = vr_tensor[:]
        torch_tensor = torch_tensor.to(th.float32)
        return torch_tensor

    def getitem(self, video_path):
        video_tensor = self.read_video(video_path)
        video = video_tensor.permute(0, 3, 1, 2)
        return video  # torch.Size([900, 3, 406, 720]) === (Number of frames, number of chanels , h, w)
