from model.setr.SETR import SETR_PUP
from model.unet.unet_model import UNet
from model.segformer import SegFormerUp


def build_model(config):
    model_type = config['model_type']
    assert model_type in ['SETR_PUP', 'UNet', 'SegFormer']
    if model_type == 'SETR_PUP':
        return SETR_PUP(
            img_dim=config['data_img_size'],
            patch_dim=16,
            num_channels=config['in_channels'],
            num_classes=config['n_classes'],
            embedding_dim=768,
            num_heads=16,
            num_layers=24,
            hidden_dim=4096,
            dropout_rate=0.1,
            attn_dropout_rate=0.1
        )
    if model_type == 'UNet':
        return UNet(n_channels=config['in_channels'],
                    n_classes=config['n_classes'])
    if model_type == 'SegFormer':
        return SegFormerUp(
            dims=(32, 64, 160, config['data_img_size']),
            heads=(1, 2, 5, 8),
            ff_expansion=(8, 8, 4, 4),
            reduction_ratio=(8, 4, 2, 1),
            num_layers=2,
            channels=config['in_channels'],
            decoder_dim=config['data_img_size'],
            num_classes=config['n_classes']
        )
