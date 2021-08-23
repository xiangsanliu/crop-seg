import model.segformer.mix_transformer as encoders
from model.deeplabv3plus.resnet import build_resnet50
from .segformer_decoder import SegFormerHead
import torch
import torch.nn as nn
import model.deeplabv3plus.resnet as Backbones


class Segformer(nn.Module):
    def __init__(self, encode_config, decoder_config):
        super().__init__()
        self.encoder = getattr(encoders, encode_config["type"])()
        self.encoder.load_pretrained(encode_config["pretrained"])
        self.decoder = SegFormerHead(**decoder_config)
        self.upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.upsample(x)
        return x


class HybridSegformer(nn.Module):
    def __init__(self, encode_config, decoder_config):
        super().__init__()
        self.encoder = getattr(encoders, encode_config["type"])()
        self.encoder.load_pretrained(encode_config["pretrained"])

        self.resnet_config = encode_config["resnet_config"]
        self.resnet = resnet50(**self.resnet_config)

        self.decoder = SegFormerHead(**decoder_config)
        self.fuse1 = Fuse2d(256, 64)
        self.fuse2 = Fuse2d(64, 5)

    def forward(self, x):
        stage1, stage2 = self.resnet(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.cat([stage2, x], dim=1)
        x = self.fuse1(x)
        x = torch.cat([stage1, x], dim=1)
        x = self.fuse2(x)
        return x

    def _build_resnet(self):
        resnet_type = self.resnet_config.pop("type")
        return getattr(Backbones, resnet_type)(**self.resnet_config)


class Fuse2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Fuse2d, self).__init__()
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x):
        x = self.linear_fuse(x)
        x = self.upsample(x)
        return x


class resnet50(nn.Module):
    def __init__(self, pretrained=True, progress=True, **kwargs):
        super(resnet50, self).__init__()
        backbone = build_resnet50(pretrained=pretrained, progress=progress, **kwargs)

        backbones = list(backbone.children())
        self.stage1 = nn.Sequential(*backbones[0:3])
        self.stage2 = nn.Sequential(*backbones[3:5])

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        return x1, x2
