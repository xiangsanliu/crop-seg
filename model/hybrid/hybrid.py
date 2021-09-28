import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from .single_transformer import mit_b4

def get_backbone(name, pretrained=True):
    if name == "resnet18":
        backbone = models.resnet18(pretrained=pretrained)
    elif name == "resnet34":
        backbone = models.resnet34(pretrained=pretrained)
    elif name == "resnet50":
        backbone = models.resnet50(pretrained=pretrained)
    elif name == "resnet101":
        backbone = models.resnet101(pretrained=pretrained)
    elif name == "resnet152":
        backbone = models.resnet152(pretrained=pretrained)
    feature_names = ["relu", "layer1", "layer2", "layer3"]
    backbone_output = "layer4"

    return backbone, feature_names, backbone_output

class UpConv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x
    


class Hybrid(nn.Module):
    def __init__(
        self,
        backbone_name="resnet50",
        pretrained=True,
        decoder_filters=(256, 128, 64, 32, 16),
    ):
        super(Hybrid, self).__init__()

        self.backbone, self.shortcut_features, self.bb_out_name = get_backbone(
            backbone_name, pretrained=pretrained
        )
        shortcut_chs, bb_out_chs = [0, 64, 256, 512, 1024], 2048
        decoder_filters = decoder_filters[: len(self.shortcut_features)]
        self.segformer = mit_b4()
        self.up1 = UpConv(2048, 1024)
    
    def forward(self, x):
        x = self.segformer(x)
        # x, features = self._forword_backbone(x)
        # for i in features:
            # print(i.shape)
        return x
    
    def _forword_backbone(self, x):
        features = []
        print(self.shortcut_features)
        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                features.append(x)
                print(name, x.shape)
            if name == self.bb_out_name:
                break
        
        return x, reversed(features)
