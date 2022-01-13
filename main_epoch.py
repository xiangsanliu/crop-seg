import torch
import json
from tqdm import tqdm

from pprint import pformat
from utils.my_logging import Logger
from data.dataloader import build_dataloader
from model import build_model
from argparse import ArgumentParser
from utils.metrics import runningScore
from model.loss import LabelSmoothingCrossEntropy2d
import numpy as np
import random
import configs
import time
from PIL import Image


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(666)


class Trainer(object):
    def __init__(self, args):
        config_file = args.config
        config_file = config_file.replace(".py", "").replace("configs/", "")
        config = getattr(configs, config_file)
        self.config = config
        self.logger = Logger(config_file)
        self.model = build_model(config["model"])
        self._unpack_train_config(args)
        self.train_pipeline = config["train_pipeline"]
        self.test_pipeline = config["test_pipeline"]
        self.best_f1 = 0
        self.best_iou = 0
        self.step_list, self.mf1_list, self.mIoU_list = [], [], []
        self.model.to(self.device)
        self.logger.info(f"\n{args}")

    def train(self):
        self.logger.info(f"\n{json.dumps(self.config, indent=4)}")
        self.loss_func = torch.nn.CrossEntropyLoss(reduction="mean")
        self.train_loader = build_dataloader(self.train_pipeline)
        optimizer = torch.optim.Adam(
            [{"params": self.model.parameters(), "initial_lr": self.lr}],
            lr=self.lr,
            weight_decay=1e-5,
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.5,
            last_epoch=self.last_epoch,
        )
        loss_list = []
        epoch_list = []
        for epoch in range(self.total_epochs):
            self.logger.info(
                f"Epoch:{epoch+1}/{self.total_epochs}, lr={lr_scheduler.get_lr()}. Training..."
            )
            report_loss = 0.0
            for img, mask in tqdm(self.train_loader, desc=f"Train", ncols=100, mininterval=300):
                optimizer.zero_grad()
                img = img.to(self.device)
                mask = mask.to(self.device)
                pred = self.model(img)
                loss = self.loss_func(pred, mask)
                loss.backward()
                optimizer.step()
                report_loss += loss.item()
            self.logger.info(f"Epoch:{epoch+1}/{self.total_epochs}, loss={report_loss/len(self.train_loader)}.")
            loss_list.append(report_loss / len(self.train_loader))
            epoch_list.append(epoch)
            lr_scheduler.step()
            self.logger.save_model(self.model)
            self.logger.plot_loss(epoch_list=epoch_list, loss_list=loss_list)
        self.logger.log_finish(0)

    def eval(self):
        self.model.eval()
        self.model.load_state_dict(torch.load(self.weight, map_location="cpu"))
        self.test_loader = build_dataloader(self.test_pipeline)
        running_metrics_val = runningScore(self.n_classes)
        with torch.no_grad():
            for val_img, val_mask in tqdm(self.test_loader, desc="Valid", ncols=100):
                val_img = val_img.to(self.device)

                pred_img_1 = self.model(val_img)

                # pred_img_2 = self.model(torch.flip(val_img, [-1]))
                # pred_img_2 = torch.flip(pred_img_2, [-1])

                # pred_img_3 = self.model(torch.flip(val_img, [-2]))
                # pred_img_3 = torch.flip(pred_img_3, [-2])

                # pred_img_4 = self.model(torch.flip(val_img, [-1, -2]))
                # pred_img_4 = torch.flip(pred_img_4, [-1, -2])

                # pred_list = pred_img_1 + pred_img_2 + pred_img_3 + pred_img_4
                pred_list = pred_img_1
                pred_list = torch.argmax(pred_list.cpu(), 1).byte().numpy()
                gt = val_mask.data.numpy()
                # vis_label(pred_list, gt)
                running_metrics_val.update(gt, pred_list)
        score, mean_f1, mIoU = running_metrics_val.get_scores()
        self.mf1_list.append(mean_f1)
        self.mIoU_list.append(mIoU)
        self.logger.info(f"---------------------------------------")
        for k, v in score.items():
            self.logger.info(f"{k}{v}")
        # if mean_f1 > self.best_f1:
        #     self.logger.save_model(self.model)
        #     self.best_f1 = mean_f1
        #     self.best_iou = mIoU
        #     self.early_stopping = 0
        # else:
        #     self.early_stopping += 1
        # self.model.train()
        return score, mean_f1, mIoU

    def _unpack_train_config(self, args):
        self.lr = args.lr
        self.total_epochs = args.total_epochs
        self.last_epoch = args.last_epoch
        self.n_classes = args.n_classes
        self.early_stopping = args.early_stopping
        self.device = args.device
        self.weight = args.weight
        self.with_eval = args.with_eval

        if args.resume:
            self.model.load_state_dict(torch.load(args.resume, map_location="cpu"))
            print(f"Restored from {self.last_step} step!")
            self.logger.info(
                f"Restored from {self.last_step} step, model:{args.resume}"
            )


def vis_label(pred, gt):
    b, _, _ = gt.shape
    for i in range(b):
        pred_vis = convert_label(pred[i])
        gt_vis = convert_label(gt[i])
        # name = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        name = time.time()
        result = np.concatenate((pred_vis, gt_vis), axis=1)
        # print("np.unique: ", np.unique(result))
        target_path = f"vis/{name}.png"
        result = Image.fromarray(np.uint8(result))
        result.save(target_path)


def convert_label(label):
    label = np.asarray(label)
    R = label.copy()  # 红色通道
    R[R == 1] = 0
    R[R == 2] = 0
    R[R == 3] = 255
    R[R == 4] = 127
    G = label.copy()  # 绿色通道
    G[G == 1] = 0
    G[G == 2] = 255
    G[G == 3] = 0
    R[G == 4] = 127
    B = label.copy()  # 蓝色通道
    B[B == 1] = 255
    B[B == 2] = 0
    B[B == 3] = 0
    R[B == 4] = 127
    return np.dstack((R, G, B))


def parse_args():
    parser = ArgumentParser(description="Training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n_classes", type=int, required=True)
    parser.add_argument("--total_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--last_epoch", type=int, default=0)
    parser.add_argument("--early_stopping", type=int, default=0)
    parser.add_argument("--do_eval", type=bool, default=False)
    parser.add_argument("--with_eval", type=bool, default=False)
    parser.add_argument("--weight", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    if args.do_eval:
        trainer.eval()
    else:
        trainer.train()
