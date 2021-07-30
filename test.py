
# from model.setr.SETR import SETR_Naive, SETR_PUP
from segformer_pytorch import Segformer
from swin_transformer_pytorch import SwinTransformer
from model.segformer import SegFormerUp

from data.dataloader import build_dataloader
from model import build_model
from configs.segformer_tianchi import config
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

def test_dataset():
    train_pipeline = dict(
        dataloader = dict(batch_size = 16,num_workers = 12,drop_last = True,pin_memory=True,shuffle=True),

        dataset = dict(type="PNG_Dataset",
                    csv_file=r'/home/xiangjianjian/Projects/spectral-setr/dataset/tianchi/round1/train.csv',
                    image_dir=r'/home/xiangjianjian/Projects/spectral-setr/dataset/tianchi/round1/image',
                    mask_dir=r'/home/xiangjianjian/Projects/spectral-setr/dataset/tianchi/round1/label'),

        transforms = [
            dict(type="RandomCrop",p=1,output_size=(512,512)),
            dict(type="RandomHorizontalFlip",p=0.5),
            dict(type="RandomVerticalFlip",p=0.5),
            # dict(type="ColorJitter",brightness=0.08,contrast=0.08,saturation=0.08,hue=0.08),
            dict(type="ToTensor",),
            dict(type="Normalize",mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],inplace=True),
            ],
    )
    train_loader = build_dataloader(train_pipeline)
    model = build_model(config['model'])
    print(model)

def test_segformer():
    

if __name__ == '__main__':
    # main()
    test_dataset()
    pass
