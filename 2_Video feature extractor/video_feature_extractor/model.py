import sys
import torch as th
import torchvision.models as models
from videocnn.models import resnext
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights, ResNet152_Weights



class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        return th.mean(x, dim=[-2, -1])


def get_model(args):
    assert args.type in ['2d', '3d']
    if args.type == '2d':
        print('Loading 2D-ResNet-152 ...')
        model = models.resnet152(weights = ResNet152_Weights.DEFAULT)
        model = nn.Sequential(*list(model.children())[:-2], GlobalAvgPool())
        #model = model.cuda()
        model = model.cpu()
    else:
        print('Loading 3D-ResneXt-101 ...')
        model = resnext.resnet101(
            num_classes=400,
            shortcut_type='B',
            cardinality=32,
            sample_size=112,
            sample_duration=16,
            last_fc=False)
        #model = model.cuda()
        model = model.cpu()
        model_data = th.load(args.resnext101_model_path)
        model.load_state_dict(model_data)

    model.eval()
    print('loaded')
    return model
