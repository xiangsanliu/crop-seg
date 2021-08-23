import torch
import torch.nn as nn
from tqdm import tqdm


from utils.my_logging import Logger
from data.dataloader import build_dataloader
from model import build_model
from utils import parse_args
from model.loss import  LabelSmoothingCrossEntropy2d
import configs


def train(config_file):
    config = getattr(configs, config_file)
    logger = Logger(config_file)
    model = build_model(config["model"])
    logger.info(model)
    loss_func = LabelSmoothingCrossEntropy2d()
    train_loader = build_dataloader(config["train_pipeline"])
    train_config = config["train_config"]
    lr_scheduler_config = config["lr_scheduler"]
    device = train_config["device"]
    last_epoch = 0
    if train_config["restore"]:
        model.load_state_dict(
            torch.load(train_config["model_save_path"], map_location="cpu")
        )
        last_epoch = train_config["last_epoch"]
        logger.info(f"Restored from the {last_epoch} epoch")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_config["lr"], weight_decay=1e-5
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **lr_scheduler_config)
    step = 0
    report_loss = 0.0
    loss_list = []
    epoch_list = []
    model.to(device)
    epoch = last_epoch
    while epoch < train_config["epoches"]:
        epoch += 1
        logger.info(
            f"Epoch:{epoch}/{train_config['epoches']}, lr={lr_scheduler.get_lr()}"
        )
        for img, mask in tqdm(
            train_loader, total=len(train_loader), desc=f"Train:{epoch}/{train_config['epoches']}", unit=" step", ncols=0
        ):
            optimizer.zero_grad()
            step += 1
            img = img.to(device)
            mask = mask.to(device)
            pred_img = model(img)
            loss = loss_func(pred_img, mask)
            report_loss += loss.item()
            loss.backward()
            optimizer.step()
        logger.info(f"Train loss = {report_loss / step}")
        loss_list.append(report_loss / step)
        epoch_list.append(epoch)
        step = 0
        report_loss = 0.0
        lr_scheduler.step()
        logger.save_model(model)
        logger.plot_loss(epoch_list, loss_list)
    logger.log_finish()


def main():
    args = vars(parse_args())
    config_file = args["config"]

    train(config_file)


if __name__ == "__main__":
    main()
