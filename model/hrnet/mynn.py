"""
Custom Norm wrappers to enable sync BN, regular BN and for weight
initialization
"""
import re
import torch
import torch.nn as nn
from .attr_dict import AttrDict
from collections import OrderedDict
import model.hrnet.hrnetv2 as hrnetv2

align_corners = True


def initialize_weights(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(
        x, size=size, mode="bilinear", align_corners=align_corners
    )


def Upsample2(x):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(
        x, scale_factor=2, mode="bilinear", align_corners=align_corners
    )


def Down2x(x):
    return torch.nn.functional.interpolate(
        x, scale_factor=0.5, mode="bilinear", align_corners=align_corners
    )


def Up15x(x):
    return torch.nn.functional.interpolate(
        x, scale_factor=1.5, mode="bilinear", align_corners=align_corners
    )


def DownX(x, scale_factor):
    """
    scale x to the same size as y
    """
    x_scaled = torch.nn.functional.interpolate(
        x,
        scale_factor=scale_factor,
        mode="bilinear",
        align_corners=align_corners,
        recompute_scale_factor=True,
    )
    return x_scaled


def ResizeX(x, scale_factor):
    """
    scale x by some factor
    """
    x_scaled = torch.nn.functional.interpolate(
        x,
        scale_factor=scale_factor,
        mode="bilinear",
        align_corners=align_corners,
        recompute_scale_factor=True,
    )
    return x_scaled


def scale_as(x, y):
    """
    scale x to the same size as y
    """
    y_size = y.size(2), y.size(3)

    x_scaled = torch.nn.functional.interpolate(
        x, size=y_size, mode="bilinear", align_corners=align_corners
    )
    return x_scaled


def get_trunk(trunk_name, output_stride=8):
    """
    Retrieve the network trunk and channel counts.
    """
    assert output_stride == 8, "Only stride8 supported right now"

    if trunk_name == "hrnetv2":
        backbone = hrnetv2.get_seg_model()
        high_level_ch = backbone.high_level_ch
        s2_ch = -1
        s4_ch = -1
    else:
        raise "unknown backbone {}".format(trunk_name)

    return backbone, s2_ch, s4_ch, high_level_ch


def Norm2d(in_channels, **kwargs):
    """
    Custom Norm Function to allow flexible switching
    """
    normalization_layer = nn.BatchNorm2d(in_channels, **kwargs)
    return normalization_layer


def BNReLU(ch):
    return nn.Sequential(Norm2d(ch), nn.ReLU())





class ConvBnRelu(nn.Module):
    # https://github.com/lingtengqiu/Deeperlab-pytorch/blob/master/seg_opr/seg_oprs.py
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, norm_layer=Norm2d
    ):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = norm_layer(out_planes, eps=1e-5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=(6, 12, 18)):
        super(AtrousSpatialPyramidPoolingModule, self).__init__()

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise "output stride of {} not supported".format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                Norm2d(reduction_dim),
                nn.ReLU(inplace=True),
            )
        )
        # other rates
        for r in rates:
            self.features.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_dim,
                        reduction_dim,
                        kernel_size=3,
                        dilation=r,
                        padding=r,
                        bias=False,
                    ),
                    Norm2d(reduction_dim),
                    nn.ReLU(inplace=True),
                )
            )
        self.features = nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            Norm2d(reduction_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


class ASPP_edge(AtrousSpatialPyramidPoolingModule):
    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=(6, 12, 18)):
        super(ASPP_edge, self).__init__(
            in_dim=in_dim,
            reduction_dim=reduction_dim,
            output_stride=output_stride,
            rates=rates,
        )
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, reduction_dim, kernel_size=1, bias=False),
            Norm2d(reduction_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, edge):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features
        edge_features = Upsample(edge, x_size[2:])
        edge_features = self.edge_conv(edge_features)
        out = torch.cat((out, edge_features), 1)

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


def dpc_conv(in_dim, reduction_dim, dil, separable):
    if separable:
        groups = reduction_dim
    else:
        groups = 1

    return nn.Sequential(
        nn.Conv2d(
            in_dim,
            reduction_dim,
            kernel_size=3,
            dilation=dil,
            padding=dil,
            bias=False,
            groups=groups,
        ),
        nn.BatchNorm2d(reduction_dim),
        nn.ReLU(inplace=True),
    )


class DPC(nn.Module):
    """
    From: Searching for Efficient Multi-scale architectures for dense
    prediction
    """

    def __init__(
        self,
        in_dim,
        reduction_dim=256,
        output_stride=16,
        rates=[(1, 6), (18, 15), (6, 21), (1, 1), (6, 3)],
        dropout=False,
        separable=False,
    ):
        super(DPC, self).__init__()

        self.dropout = dropout
        if output_stride == 8:
            rates = [(2 * r[0], 2 * r[1]) for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise "output stride of {} not supported".format(output_stride)

        self.a = dpc_conv(in_dim, reduction_dim, rates[0], separable)
        self.b = dpc_conv(reduction_dim, reduction_dim, rates[1], separable)
        self.c = dpc_conv(reduction_dim, reduction_dim, rates[2], separable)
        self.d = dpc_conv(reduction_dim, reduction_dim, rates[3], separable)
        self.e = dpc_conv(reduction_dim, reduction_dim, rates[4], separable)

        self.drop = nn.Dropout(p=0.1)

    def forward(self, x):
        a = self.a(x)
        b = self.b(a)
        c = self.c(a)
        d = self.d(a)
        e = self.e(b)
        out = torch.cat((a, b, c, d, e), 1)
        if self.dropout:
            out = self.drop(out)
        return out


def get_aspp(high_level_ch, bottleneck_ch, output_stride, dpc=False):
    """
    Create aspp block
    """
    if dpc:
        aspp = DPC(high_level_ch, bottleneck_ch, output_stride=output_stride)
    else:
        aspp = AtrousSpatialPyramidPoolingModule(
            high_level_ch, bottleneck_ch, output_stride=output_stride
        )
    aspp_out_ch = 5 * bottleneck_ch
    return aspp, aspp_out_ch


def BNReLU(ch):
    return nn.Sequential(Norm2d(ch), nn.ReLU())


def make_seg_head(in_ch, out_ch):
    bot_ch = 256
    return nn.Sequential(
        nn.Conv2d(in_ch, bot_ch, kernel_size=3, padding=1, bias=False),
        Norm2d(bot_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(bot_ch, bot_ch, kernel_size=3, padding=1, bias=False),
        Norm2d(bot_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(bot_ch, out_ch, kernel_size=1, bias=False),
    )


def init_attn(m):
    for module in m.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.5)
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(1)
            module.bias.data.zero_()


def make_attn_head(in_ch, out_ch):
    bot_ch = 256

    od = OrderedDict(
        [
            ("conv0", nn.Conv2d(in_ch, bot_ch, kernel_size=3, padding=1, bias=False)),
            ("bn0", Norm2d(bot_ch)),
            ("re0", nn.ReLU(inplace=True)),
        ]
    )

    od["conv1"] = nn.Conv2d(bot_ch, bot_ch, kernel_size=3, padding=1, bias=False)
    od["bn1"] = Norm2d(bot_ch)
    od["re1"] = nn.ReLU(inplace=True)

    od["conv2"] = nn.Conv2d(bot_ch, out_ch, kernel_size=1, bias=False)
    od["sig"] = nn.Sigmoid()

    attn_head = nn.Sequential(od)
    # init_attn(attn_head)
    return attn_head


def fmt_scale(prefix, scale):
    """
    format scale name
    :prefix: a string that is the beginning of the field name
    :scale: a scale value (0.25, 0.5, 1.0, 2.0)
    """

    scale_str = str(float(scale))
    scale_str.replace(".", "")
    return f"{prefix}_{scale_str}x"
