import torch
import torch.nn as nn
from tqdm import tqdm


from utils.model_tools import ModelValidator
from utils.my_logging import Logger
from data.dataloader import build_dataloader
from model import build_model
from utils import parse_args
import configs


config = getattr(configs, 'hybrid_segformer_tianchi_2')
model = build_model(config["model"])
x = torch.randn(8, 3, 512, 512)
x = model(x)
print(x.shape)