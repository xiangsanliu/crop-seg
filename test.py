import torch
import torch.nn as nn
from tqdm import tqdm

from utils.model_tools import ModelValidator
from utils.my_logging import Logger
from data.dataloader import build_dataloader
from model import build_model
from utils import parse_args
import configs


def test(config_file, weight_file):
    config = getattr(configs, config_file)
    model = build_model(config["model"])
    loss_func = nn.CrossEntropyLoss()
    test_loader = build_dataloader(config["test_pipeline"])
    train_config = config["train_config"]
    device = train_config["device"]
    model.to(device)
    validator = ModelValidator(train_config, loss_func)
    model.load_state_dict(torch.load(weight_file, map_location="cpu"))
    validator.validate_model(model, test_loader, device)
    pass


if __name__ == "__main__":
    args = vars(parse_args())
    config_file = args["config"]
    weight_file = args["weight"]
    test(config_file, weight_file)
    pass
