from .segformer_pytorch import Segformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class SegFormerUp(nn.Module):
    def __init__(
        self,
        dims,
        heads,
        ff_expansion,
        reduction_ratio,
        num_layers,
        channels,
        decoder_dim,
        num_classes,
    ):
        super(SegFormerUp, self).__init__()
        self.segformer = Segformer(
            dims=dims,
            heads=heads,
            ff_expansion=ff_expansion,
            reduction_ratio=reduction_ratio,
            num_layers=num_layers,
            channels=channels,
            decoder_dim=decoder_dim,
            num_classes=num_classes,
        )
        self.upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

    def forward(self, x):
        x = self.segformer(x)
        x = self.upsample(x)
        return x
