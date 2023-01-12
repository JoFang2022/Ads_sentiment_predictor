import torch as th
from torchvision import transforms


class Normalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = th.FloatTensor(mean).view(1, 3, 1, 1)
        self.std = th.FloatTensor(std).view(1, 3, 1, 1)
        self.inplace = inplace

    def __call__(self, tensor):
        tensor = (tensor - self.mean) / (self.std + 1e-8)
        return tensor


class Preprocessing(object):
    def __init__(self):
        self.norm = transforms.Normalize(mean=[110.6, 103.2, 96.3], std=[1.0, 1.0, 1.0])
        self.crop = transforms.CenterCrop((112, 112))
        self.resize = transforms.Resize(112)

    def _zero_pad(self, tensor, size):
        n = size - len(tensor) % size
        if n == size:
            return tensor
        else:
            z = th.zeros(n, tensor.shape[1], tensor.shape[2], tensor.shape[3])
            return th.cat((tensor, z), 0)

    def __call__(self, tensor):
        tensor = self._zero_pad(tensor, 16)
        tensor = self.norm(tensor)
        tensor = self.resize(tensor)
        tensor = self.crop(tensor)
        tensor = tensor.view(-1, 16, 3, 112, 112)
        tensor = tensor.transpose(1, 2)
        return tensor
