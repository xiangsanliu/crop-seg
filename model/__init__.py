from model.segformer import Segformer, HybridSegformer, IndicesFormer
from model.deeplabv3plus import DeepLabV3Plus
from model.unet import Unet
from model.beit import BEiT
from model.hrnet.ocrnet import HRNet_Mscale
from model.hrnet.rmi import RMILoss
import model as models
import model.loss as Losses
from copy import deepcopy

# models = {
#     "hrnet": HRNet_Mscale,
# }


def build_model(config):
    config = deepcopy(config)
    model_type = config["type"]
    model_config = config["model_config"]
    if model_type == "hrnet":
        criterion = RMILoss(num_classes=2)
        model = HRNet_Mscale(num_classes=2, criterion=criterion)
        model.load_pretrained("pretrained/hrnet.pth")
        return model
    model = getattr(models, model_type)(**model_config)
    return model


def build_loss(config):
    config = deepcopy(config)
    loss_type = config.pop("type")
    return getattr(Losses, loss_type)(**config)
