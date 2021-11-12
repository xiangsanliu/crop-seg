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
import numpy as np
import random
import configs

device = "cuda" if torch.cuda.is_available() else "cpu"


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(20)


class Trainer(object):
    def __init__(self, config_file):
        config = getattr(configs, config_file)
        self.config = config
        self.logger = Logger(config_file)
        train_config = config["train_config"]
        self.model = build_model(config["model"])
        self._unpack_train_config(train_config)
        self.loss_func = torch.nn.CrossEntropyLoss(reduction="mean")
        self.train_loader = build_dataloader(config["train_pipeline"])
        self.test_loader = build_dataloader(config["test_pipeline"])
        self.best_f1 = 0
        self.best_iou = 0
        self.early_stopping = 0
        self.step_list, self.mf1_list, self.mIoU_list = [], [], []

    def train(self):
        self.logger.info(f"\n{json.dumps(self.config, indent=4)}")
        optimizer = torch.optim.Adam(
            [{"params": self.model.parameters(), "initial_lr": self.lr}],
            lr=self.lr,
            weight_decay=1e-5,
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.5,
            last_epoch=self.last_step // self.eval_steps,
        )
        self.model.to(self.device)
        train_iter = iter(self.train_loader)
        report_loss = 0
        for step in tqdm(
            range(self.last_step, self.total_steps), desc=f"Train", ncols=100
        ):

            try:
                img, mask = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                img, mask = next(train_iter)

            optimizer.zero_grad()
            img = img.to(device)
            mask = mask.to(device)
            inputs = {"images": img, "gts": mask}
            loss = self.model(inputs)
            report_loss += loss.item()
            loss.backward()
            optimizer.step()

            # eval
            if (step + 1) % self.eval_steps == 0:
                self.eval()
                lr_scheduler.step()
                self.step_list.append(step + 1)
                self.logger.plot_acc(self.step_list, self.mIoU_list, self.mf1_list)
                if self.early_stopping > 10:
                    break
        self.logger.log_finish(self.best_iou)

    def eval(self):
        self.model.eval()

        running_metrics_val = runningScore(self.n_classes)
        with torch.no_grad():
            for val_img, val_mask in tqdm(self.test_loader, desc="Valid", ncols=100):
                val_img = val_img.to(device)

                pred_img_1 = self.model({"images": val_img})

                # pred_img_2 = self.model(torch.flip(val_img, [-1]))
                # pred_img_2 = torch.flip(pred_img_2, [-1])

                # pred_img_3 = self.model(torch.flip(val_img, [-2]))
                # pred_img_3 = torch.flip(pred_img_3, [-2])

                # pred_img_4 = self.model(torch.flip(val_img, [-1, -2]))
                # pred_img_4 = torch.flip(pred_img_4, [-1, -2])

                # pred_list = pred_img_1 + pred_img_2 + pred_img_3 + pred_img_4
                pred_list = pred_img_1["pred"]
                pred_list = torch.argmax(pred_list.cpu(), 1).byte().numpy()
                gt = val_mask.data.numpy()
                running_metrics_val.update(gt, pred_list)
        score, mean_f1, mIoU = running_metrics_val.get_scores()
        self.mf1_list.append(mean_f1)
        self.mIoU_list.append(mIoU)
        self.logger.info(f"---------------------------------------")
        for k, v in score.items():
            self.logger.info(f"{k}{v}")
        if mean_f1 > self.best_f1:
            self.logger.save_model(self.model)
            self.best_f1 = mean_f1
            self.best_iou = mIoU
            self.early_stopping = 0
        else:
            self.early_stopping += 1
        self.model.train()
        return score, mean_f1, mIoU

    def _unpack_train_config(self, train_config):
        self.device = train_config["device"]
        self.lr = train_config["lr"]
        self.total_steps = train_config["total_steps"]
        self.eval_steps = train_config["eval_steps"]
        self.last_step = 0
        self.restore = train_config["restore"]
        self.restore_path = train_config["restore_path"]
        self.n_classes = train_config["n_classes"]
        if self.restore:

            self.last_step = train_config["last_step"]
            self._restore()

    def _restore(self):
        self.model.load_state_dict(torch.load(self.restore_path, map_location="cpu"))
        print(f"Restored from {self.last_step} step!")
        self.logger.info(
            f"Restored from {self.last_step} step, model:{self.restore_path}"
        )


if __name__ == "__main__":
    args = vars(parse_args())
    config_file = args["config"]
    config_file = config_file.replace(".py", "").replace("configs/", "")
    trainer = Trainer(config_file)
    trainer.train()
