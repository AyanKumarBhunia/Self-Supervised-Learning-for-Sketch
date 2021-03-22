import torch.nn as nn
import torchvision.models as backbone_
import torch.nn.functional as F
import torch
from torchvision.ops import MultiScaleRoIAlign
from collections import OrderedDict
import torch
from torch.nn.utils.rnn import pad_sequence
import torchvision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Resnet_Network(nn.Module):
    def __init__(self, hp, num_class = 4):
        super(Resnet_Network, self).__init__()
        backbone = backbone_.resnet50(pretrained=False) #resnet50, resnet18, resnet34

        self.features = nn.Sequential()
        for name, module in backbone.named_children():
            if name not in ['avgpool', 'fc']:
                self.features.add_module(name, module)

        self.pool_method =  nn.AdaptiveMaxPool2d(1) # as default
        self.classifier = nn.Linear(2048, num_class)

    def forward(self, x):
        x = self.features(x)
        x = self.pool_method(x).view(-1, 2048)
        x = self.classifier(x)
        return x

    def extract_features(self, x, every_layer=True):
        feature_list = {}
        batch_size = x.shape[0]
        # https://stackoverflow.com/questions/47260715/
        # how-to-get-the-value-of-a-feature-in-a-layer-that-match-a-the-state-dict-in-pyto
        for name, module in self.features._modules.items():
            x = module(x)
            if every_layer and name in ['layer1', 'layer2', 'layer3', 'layer4']:
                feature_list[name] = self.pool_method(x).view(batch_size, -1)

        if not feature_list:
            feature_list['pre_logits'] = self.pool_method(x).view(batch_size, -1)

        return feature_list


