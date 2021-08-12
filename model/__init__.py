from model.segformer import Segformer, HybridSegformer
from model.deeplabv3plus import DeepLabV3Plus
import model as models
import model.loss as Losses
from copy import deepcopy


def build_model(config):
    config = deepcopy(config)
    model_type = config['type']
    model_config = config['model_config']
    model = getattr(models, model_type)(**model_config)
    return model


def build_loss(config):
    config = deepcopy(config)
    loss_type = config.pop('type')
    return getattr(Losses, loss_type)(**config)
