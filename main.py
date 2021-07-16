from trans_config import TransConfig
from encoder import PatchEmbedding, PositionEmbeding
import torch

config = TransConfig(
    patch_size=(32, 32),
    in_channel=3,
    out_channel=1,
    embed_dim=1024,
    num_hidden_layers=8,
    num_attention_heads=16,
    sample_rate=4
)


def main():
    net = PatchEmbedding(config)
    pos = PositionEmbeding(config)
    t1 = torch.rand(2, 3, 512, 512)
    t1 = net(t1)
    print(t1.shape)


if __name__ == '__main__':
    main()
