import torch as th
from videocnn.models import resnext
from torch import nn


class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        return th.mean(x, dim=[-2, -1])


def get_model():
    model_path = "./model/resnext101.pth"
    print("Loading 3D-ResneXt-101 ...")
    model = resnext.resnet101(
        num_classes=400,
        shortcut_type="B",
        cardinality=32,
        sample_size=112,
        sample_duration=16,
        last_fc=False,
    )
    # model = model.cuda()
    model = model.cpu()
    model_data = th.load(model_path)
    model.load_state_dict(model_data)
    model.eval()
    print("loaded")
    return model
