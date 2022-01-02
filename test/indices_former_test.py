import torch
import torch.nn as nn
from tqdm import tqdm


from model import build_model
import configs
from model import build_model


config = getattr(configs, 'indicesformer_b4_tianchi')
# config = getattr(configs, 'unet_tianchi2_label_no_overlap')
model = build_model(config["model"])
x = torch.randn(2, 3, 512, 512)
x = model(x)
print(x.shape)