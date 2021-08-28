import torch
import torch.nn as nn
from utils.model_tools import ModelValidator
from utils.my_logging import Logger
from data.dataloader import build_dataloader
from model import build_model
from utils import parse_args
import configs


def test(config_file, weight_file):
    config = getattr(configs, config_file)
    model = build_model(config["model"])
    logger = Logger(config_file, 'test')
    loss_func = nn.CrossEntropyLoss()
    test_loader = build_dataloader(config["test_pipeline"])
    train_config = config["train_config"]
    device = train_config["device"]
    model.to(device)
    validator = ModelValidator(train_config, loss_func)
    model.load_state_dict(torch.load(weight_file, map_location="cpu"))
    logger.info(f"config: {config_file}, weight: {weight_file}")
    score, loss = validator.validate_model(model, test_loader, device)
    
    for k, v in score.items():
        logger.info(f"{k}{v}")


if __name__ == "__main__":
    args = vars(parse_args())
    config_file = args["config"]
    config_file = config_file.replace(".py", "").replace("configs/", "")
    weight_file = args["weight"]
    test(config_file, weight_file)
    pass
