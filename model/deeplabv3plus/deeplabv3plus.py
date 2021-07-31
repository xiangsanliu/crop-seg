import torch
import torch.nn as nn
import torch.nn.functional as F
import model.deeplabv3plus.resnet as Backbones


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes, backbone_config, head_config):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = self._build_backbone(backbone_config)
        self.head = ASPP(**head_config)

        self.out1 = nn.Sequential(nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=1, stride=1), nn.ReLU())
        self.dropout1 = nn.Dropout(0.5)
        self.up4 = nn.Upsample(scale_factor=4)
        self.up2 = nn.Upsample(scale_factor=2)

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(2048, 256, 1, bias=False), nn.ReLU())
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(512, self.num_classes, 1), nn.ReLU())
        self.dec_conv = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())

    def forward(self, x, targets=None):
        x = self.backbone(x)
        out1 = self.head(x)
        out1 = self.out1(out1)
        out1 = self.dropout1(out1)
        out1 = self.up4(out1)

        dec = self.conv1x1(x)
        dec = self.dec_conv(dec)
        dec = self.up4(dec)

        contact = torch.cat((out1, dec), dim=1)
        out = self.conv3x3(contact)
        out = self.up4(out)

        # if self.training:
            # loss = self.loss(out, targets)
            # return loss
        # else:
        return out

    def _build_backbone(self,backbone_config):
        backbone_type = backbone_config.pop('type')
        return getattr(Backbones, backbone_type)(**backbone_config)




class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, dilation_list=[6, 12, 18]):
        super(ASPP, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1,
                      padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                      stride=1, padding=dilation_list[0], dilation=dilation_list[0], bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                      stride=1, padding=dilation_list[1], dilation=dilation_list[1], bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                      stride=1, padding=dilation_list[2], dilation=dilation_list[2], bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.adapool = nn.AdaptiveAvgPool2d(1)

        self.convf = nn.Sequential(
            nn.Conv2d(in_channels=out_channels * 5, out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x))
        x3 = self.relu(self.conv3(x))
        x4 = self.relu(self.conv4(x))
        x5 = self.relu(self.conv5(self.adapool(x)))
        x5 = F.interpolate(x5, size=tuple(x4.shape[-2:]), mode='bilinear')
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # channels first
        x = self.relu(self.convf(x))
        return x
