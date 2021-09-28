import torch
import torch.nn as nn
from tqdm import tqdm


from utils.model_tools import ModelValidator
from utils.my_logging import Logger
from data.dataloader import build_dataloader
from model import build_model
from utils import parse_args
import configs
from model.hybrid.hybrid import Hybrid


config = getattr(configs, 'hybrid_b4_tianchi2_label_no_overlap')
# config = getattr(configs, 'unet_tianchi2_label_no_overlap')
model = build_model(config["model"])
x = torch.randn(2, 3, 512, 512)
x = model(x)
print(x.shape)