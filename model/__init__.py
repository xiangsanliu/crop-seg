from .segformer import *
from .setr import SETR_Naive, SETR_PUP, SETR_MLA
from .unet import UNet
import model as models
from copy import deepcopy


def build_model(config):
    config = deepcopy(config)
    model_type = config['type']
    model_config = config['model_config']
    model = getattr(models, model_type)(**model_config)
    return model