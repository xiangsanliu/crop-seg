
from setr.SETR import SETRModel
from setr.trans_config import TransConfig
import torch
from torchsummary import summary

config = TransConfig(
    patch_size=(32, 32),
    in_channels=3,
    out_channels=1,
    embed_dim=1024,
    num_hidden_layers=1,
    num_heads=16,
    sample_rate=4,
    num_classes=150
)


def main():
    net = SETRModel(config)
    print(net)
    t1 = torch.rand(2, 3, 512, 512)
    t1 = net(t1)
    print(t1.shape)

if __name__ == '__main__':
    main()