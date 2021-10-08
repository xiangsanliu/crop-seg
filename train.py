import torch
import json
from tqdm import tqdm

from pprint import pformat
from utils.my_logging import Logger
from data.dataloader import build_dataloader
from model import build_model
from utils import parse_args
from utils.metrics import runningScore
from model.loss import LabelSmoothingCrossEntropy2d
import configs

device = "cuda" if torch.cuda.is_available() else "cpu"


def train(config_file):
    config = getattr(configs, config_file)
    logger = Logger(config_file)
    logger.info(f"\n{json.dumps(config, indent=4)}")
    model = build_model(config["model"])
    loss_func = torch.nn.CrossEntropyLoss(reduction="mean")
    train_loader = build_dataloader(config["train_pipeline"])
    test_loader = build_dataloader(config["test_pipeline"])
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
    eval_step = 1000
    loss_list = []
    epoch_list = []
    mIoU_list = []
    mF1_list = []
    step_list = []
    model.to(device)
    epoch = last_epoch
    best_f1 = 0
    total_step = 0
    while epoch < train_config["epoches"]:
        epoch += 1
        logger.info(
            f"Epoch:{epoch}/{train_config['epoches']}, lr={lr_scheduler.get_last_lr()}"
        )
        for img, mask in tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Train:{epoch}/{train_config['epoches']}",
            unit=" step",
            ncols=100,
        ):
            optimizer.zero_grad()
            step += 1
            total_step += 1
            img = img.to(device)
            mask = mask.to(device)
            pred_img = model(img)
            loss = loss_func(pred_img, mask)
            report_loss += loss.item()
            loss.backward()
            optimizer.step()
            if total_step % eval_step == 0:
                score, mean_f1, mIoU = eval(model, val_loader=test_loader)
                mIoU_list.append(mIoU)
                mF1_list.append(mean_f1)
                step_list.append(total_step)
                logger.plot_acc(step_list, mIoU_list, mF1_list)
                if mean_f1 > best_f1:
                    logger.save_model(model)
                    for k, v in score.items():
                        logger.info(f"{k}{v}")

        logger.info(f"Train loss = {report_loss / step}")
        loss_list.append(report_loss / step)
        epoch_list.append(epoch)
        step = 0
        report_loss = 0.0
        lr_scheduler.step()

    logger.log_finish()


def eval(model, val_loader):
    model.eval()
    n = 0
    running_metrics_val = runningScore(2)
    with torch.no_grad():
        for val_img, val_mask in tqdm(
            val_loader, total=len(val_loader), desc="Valid", ncols=100
        ):
            n += 1
            val_img = val_img.to(device)

            pred_img_1 = model(val_img)

            pred_img_2 = model(torch.flip(val_img, [-1]))
            pred_img_2 = torch.flip(pred_img_2, [-1])

            pred_img_3 = model(torch.flip(val_img, [-2]))
            pred_img_3 = torch.flip(pred_img_3, [-2])

            pred_img_4 = model(torch.flip(val_img, [-1, -2]))
            pred_img_4 = torch.flip(pred_img_4, [-1, -2])

            pred_list = pred_img_1 + pred_img_2 + pred_img_3 + pred_img_4
            pred_list = torch.argmax(pred_list.cpu(), 1).byte().numpy()
            gt = val_mask.data.numpy()
            running_metrics_val.update(gt, pred_list)
    score, mean_f1, mIoU = running_metrics_val.get_scores()
    model.train()
    return score, mean_f1, mIoU


def main():
    args = vars(parse_args())
    config_file = args["config"]
    config_file = config_file.replace(".py", "").replace("configs/", "")
    train(config_file)


if __name__ == "__main__":
    main()
