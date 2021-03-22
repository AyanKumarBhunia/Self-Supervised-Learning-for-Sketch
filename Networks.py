import torch.nn as nn
import torchvision.models as backbone_
import torch.nn.functional as F
import torch
from torchvision.ops import MultiScaleRoIAlign
from collections import OrderedDict
import torch
import torchvision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

class Resnet_Network(nn.Module):
    def __init__(self, hp, num_class = 81):
        super(Resnet_Network, self).__init__()

        self.hp = hp
        backbone = backbone_.resnet50(pretrained=False) #resnet50, resnet18, resnet34

        self.features = nn.Sequential()
        for name, module in backbone.named_children():
            if name not in ['avgpool', 'fc']:
                self.features.add_module(name, module)

        self.pool_method =  nn.AdaptiveMaxPool2d(1) # as default

        if hp.fullysupervised:
            if hp.dataset_name == 'TUBerlin':
                num_class = 250
            elif hp.dataset_name == 'QuickDraw':
                num_class = 345
            self.classifier = nn.Linear(2048, num_class)


    def forward(self, x):
        x = self.features(x)
        x = self.pool_method(x).view(-1, 2048)
        if self.hp.fullysupervised:
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



class UNet_Decoder(nn.Module):
    def __init__(self, out_channels=3):
        super(UNet_Decoder, self).__init__()
        # self.linear_1 = nn.Linear(512, 8*8*256)
        # self.dropout = nn.Dropout(0.5)
        self.deconv_1 = Unet_UpBlock(512, 512)
        self.deconv_2 = Unet_UpBlock(512, 512)
        self.deconv_3 = Unet_UpBlock(512, 512)
        self.deconv_4 = Unet_UpBlock(512, 256)
        self.deconv_5= Unet_UpBlock(256, 128)
        self.deconv_6 = Unet_UpBlock(128, 64)
        self.deconv_7 = Unet_UpBlock(64, 32)
        self.final_image = nn.Sequential(*[nn.ConvTranspose2d(32, out_channels,
                                        kernel_size=4, stride=2,
                                        padding=1), nn.Tanh()])

    def forward(self, x):
        # x = self.linear_1(x)
        x = x.view(-1, 512, 1, 1)
        # x = self.dropout(x)
        x = self.deconv_1(x) #2
        x = self.deconv_2(x) #4
        x = self.deconv_3(x) #8
        x = self.deconv_4(x) #16
        x = self.deconv_5(x) #32
        x = self.deconv_6(x) #64
        x = self.deconv_7(x) #128
        x = self.final_image(x) #256
        return x


class Unet_UpBlock(nn.Module):
    def __init__(self, inner_nc, outer_nc):
        super(Unet_UpBlock, self).__init__()
        layers = [
            nn.ConvTranspose2d(inner_nc, outer_nc, 4, 2, 1, bias=True),
            nn.InstanceNorm2d(outer_nc),
            nn.ReLU(inplace=True),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Residual_UpBlock(nn.Module):
    def __init__(self, c_in, c_out, stride, output_padding, norm = 'InstanceNorm2d', c_hidden=None):
        super(Residual_UpBlock, self).__init__()
        c_hidden = c_out if c_hidden is None else c_hidden

        if norm == 'BatchNorm2d':
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = nn.InstanceNorm2d

        self.conv1 = nn.Sequential(
                norm_layer(c_in, affine=True),
                nn.LeakyReLU(),
                nn.Conv2d(c_in, c_hidden, kernel_size=3, stride=1, padding=1))

        self.conv2 = nn.Sequential(
                norm_layer(c_hidden, affine=True),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(c_hidden, c_out, kernel_size=3,
                stride=stride, padding=1, output_padding=output_padding))

        self.residual = nn.ConvTranspose2d(c_in, c_out, kernel_size=3,
                stride=stride, padding=1, output_padding=output_padding)


    def forward(self, x):
        residual = self.residual(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        return residual + conv2


class ResNet_Decoder(nn.Module):
    def __init__(self):
        super(ResNet_Decoder, self).__init__()
        self.upblock5 = Residual_UpBlock(512, 256, (2,1), (1,0))
        self.upblock4 = Residual_UpBlock(256, 128, (2,1), (1,0))
        self.upblock3 = Residual_UpBlock(128, 64, (2,1), (1,0))
        self.upblock2 = Residual_UpBlock(64, 32, (2,2), (1,1))
        self.upblock1 = Residual_UpBlock(32, 32, (2,2), (1,1))
        self.upblock0 = Residual_UpBlock(32, 1, (1,1), (0,0))

    def forward(self, x):
        upblock5 = self.upblock5(x)
        upblock4 = self.upblock4(upblock5)
        upblock3 = self.upblock3(upblock4)
        upblock2 = self.upblock2(upblock3)
        upblock1 = self.upblock1(upblock2)
        upblock0 = self.upblock0(upblock1)
        return torch.tanh(upblock0)


class Sketch_LSTM(nn.Module):
    def __init__(self, inp_dim=5, hidden_size=512, LSTM_num_layers=2, dropout=0.5):
        super(Sketch_LSTM, self).__init__()
        self.inp_dim, self.hidden_size, self.LSTM_num_layers, self.bidirectional = inp_dim, hidden_size, LSTM_num_layers, 2
        self.LSTM_encoder = nn.LSTM(inp_dim, hidden_size,
                num_layers=LSTM_num_layers,
                dropout=dropout,
                batch_first=True, bidirectional=True)

    def forward(self, x, seq_len):
        # batch['stroke_wise_split'][:,:,:2] /= 800
        x = pack_padded_sequence(x.to(device), seq_len.to(device), batch_first=True, enforce_sorted=False)
        _ , (x_hidden, _) = self.LSTM_encoder(x.float())
        x_hidden = x_hidden.view(self.LSTM_num_layers, self.bidirectional, seq_len.shape[0], self.hidden_size)[-1].permute(1,0,2).reshape(seq_len.shape[0], -1)

        return x_hidden