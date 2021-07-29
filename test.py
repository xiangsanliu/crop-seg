
# from model.setr.SETR import SETR_Naive, SETR_PUP
from segformer_pytorch import Segformer
from swin_transformer_pytorch import SwinTransformer
from model.segformer import SegFormerUp
import torch


def main():
    model = SegFormerUp(
        dims=(32, 64, 160, 224),
        heads=(1, 2, 5, 8),
        ff_expansion=(8, 8, 4, 4),
        reduction_ratio=(8, 4, 2, 1),
        num_layers=2,
        channels=274,
        decoder_dim=224,
        num_classes=17
    )
    x = torch.randn(1, 274, 224, 224)
    x = model(x)
    # print(net)
    print(x.shape)


if __name__ == '__main__':
    main()
