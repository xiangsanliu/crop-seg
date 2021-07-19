import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._init_decoder(config)
        # # B, 1024, 32, 32 -> B, 512, 64, 64
        # self.decoder_1 = nn.Sequential(
        #     nn.Conv2d(in_channel, features[0], 3, padding=1),
        #     nn.BatchNorm2d(features[0]),
        #     nn.ReLU(inplace=True),
        #     nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # )
        # # B, 1024, 32, 32 -> B, 64, 128, 128
        # self.decoder_2 = nn.Sequential(
        #     nn.Conv2d(features[0], features[1], 3, padding=1),
        #     nn.BatchNorm2d(features[1]),
        #     nn.ReLU(inplace=True),
        #     nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # )
        # # B, 64, 128, 128 -> B, 128, 256, 256
        # self.decoder_3 = nn.Sequential(
        #     nn.Conv2d(features[1], features[2], 3, padding=1),
        #     nn.BatchNorm2d(features[2]),
        #     nn.ReLU(inplace=True),
        #     nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # )
        # # B, 128, 256, 256 -> B, 64, 512, 512
        # self.decoder_4 = nn.Sequential(
        #     nn.Conv2d(features[2], features[3], 3, padding=1),
        #     nn.BatchNorm2d(features[3]),
        #     nn.ReLU(inplace=True),
        #     nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # )

        # self.final_out = nn.Conv2d(features[-1], 150, 3, padding=1)

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(self.config.img_size[0]/self.config.patch_size[0]),
            int(self.config.img_size[0]/self.config.patch_size[0]),
            self.config.embed_dim
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def _init_decoder(self, config):
        self.conv1 = nn.Conv2d(in_channels=config.embed_dim,
                               out_channels=config.embed_dim,
                               kernel_size=1,
                               stride=1,
                               padding=self._get_padding('VALID', (1, 1))
                               )
        self.bn1 = nn.BatchNorm2d(config.embed_dim)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=config.embed_dim,
            out_channels=config.num_classes,
            kernel_size=1,
            stride=1,
            padding=self._get_padding('VALID', (1, 1))
        )
        self.upsample = nn.Upsample(
            scale_factor=config.patch_size[0], mode='bilinear', align_corners=False)

    def decode(self, x):
        x = self._reshape_output(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x

    def forward(self, x):
        return self.decode(x)
