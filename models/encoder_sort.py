"""
Backbones supported by torchvison.
"""
import torch
import torch.nn as nn
import torchvision


class Res101Encoder(nn.Module):
    """
    Resnet101 backbone from deeplabv3
    """

    def __init__(self, replace_stride_with_dilation=None, pretrained_weights='resnet101'):
        super().__init__()
        # using pretrained model's weights
        if pretrained_weights == 'deeplabv3':
            self.pretrained_weights = torch.load(
                ".../checkpoints/deeplabv3_resnet101_coco-586e9e4e.pth", map_location='cpu')
        elif pretrained_weights == 'resnet101':
            self.pretrained_weights = torch.load(".../checkpoints/resnet101-63fe2227.pth",
                                                 map_location='cpu')
        else:
            self.pretrained_weights = pretrained_weights

        _model = torchvision.models.resnet.resnet101(pretrained=False,
                                                     replace_stride_with_dilation=replace_stride_with_dilation)
        self.backbone = nn.ModuleDict()
        for dic, m in _model.named_children():
            self.backbone[dic] = m

        self.reduce1 = nn.Conv2d(1024, 512, kernel_size=1, bias=False)
        self.reduce2 = nn.Conv2d(2048, 512, kernel_size=1, bias=False)
        self.reduce1d = nn.Linear(in_features=1000, out_features=1, bias=True)
        self.reduce2d = nn.Linear(in_features=1000, out_features=1, bias=True)

        self._init_weights()

    def forward(self, x):
        features = dict()
        x = self.backbone["conv1"](x)
        x = self.backbone["bn1"](x)
        x = self.backbone["relu"](x)
        # features['down1'] = x
        x = self.backbone["maxpool"](x)
        x = self.backbone["layer1"](x)
        x = self.backbone["layer2"](x)
        x = self.backbone["layer3"](x)  # size of x is [2, 1024, 64, 64]
        features['down2'] = self.reduce1(x)  # reduce the number of channels to 512
        x = self.backbone["layer4"](x)  # size of x is [2, 2048, 32, 32]
        features['down3'] = self.reduce2(x)  # reduce the number of channels to 512

        # feature map -> avgpool -> fc -> single value
        t = self.backbone["avgpool"](x)
        t = torch.flatten(t, 1)
        t = self.backbone["fc"](t)
        t = self.reduce1d(t)
        t1 = self.backbone["avgpool"](x)
        t1 = torch.flatten(t1, 1)
        t1 = self.backbone["fc"](t1)
        t1 = self.reduce2d(t1)
        return (features, t, t1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if self.pretrained_weights is not None:
            keys = list(self.pretrained_weights.keys())
            new_dic = self.state_dict()
            new_keys = list(new_dic.keys())

            for i in range(len(keys)):
                if keys[i] in new_keys:
                    new_dic[keys[i]] = self.pretrained_weights[keys[i]]

            self.load_state_dict(new_dic)