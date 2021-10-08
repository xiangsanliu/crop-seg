from model.beit import BEiTBackbone
from model.beit import UPerHead
import torch.nn as nn


class BEiT(nn.Module):
    def __init__(self, encoder_config, decoder_config):
        super().__init__()
        pretrained = encoder_config.pop("pretrained")
        self.encoder = BEiTBackbone(**encoder_config)
        self.encoder.load_pretrained(pretrained)
        self.decoder = UPerHead(**decoder_config)
        self.upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.upsample(x)
        return x
