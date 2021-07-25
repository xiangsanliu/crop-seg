
from model.setr.SETR import SETR_Naive
import torch


def main():
    net = SETR_Naive(
        img_dim=224,
        patch_dim=16,
        num_channels=274,
        num_classes=17,
        embedding_dim=1024,
        num_heads=16,
        num_layers=24,
        hidden_dim=4096,
    )
    t1 = torch.rand(2, 274, 224, 224)
    t1 = net(t1)
    print(t1.shape)

if __name__ == '__main__':
    main()
