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
            last_epoch=self.last_step // self.eval_steps,
        )
        train_iter = iter(self.train_loader)
        report_loss = 0
        loss_list = []
        loss_step = []
        for step in tqdm(
            range(self.last_step, self.total_steps), desc=f"Train", ncols=150
        ):

            try:
                img, mask = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                img, mask = next(train_iter)

            optimizer.zero_grad()
            img = img.to(self.device)
            mask = mask.to(self.device)
            pred_img = self.model(img)
            loss = self.loss_func(pred_img, mask)
            report_loss += loss.item()
            loss.backward()
            optimizer.step()
            if (step + 1) % self.log_steps == 0:
                loss_step.append(step + 1)
                loss_list.append(report_loss / self.log_steps)
                self.logger.info(f"{step + 1} steps: {report_loss / self.log_steps:.4f}")
                report_loss = 0
                self.logger.plot_loss(loss_step, loss_list)
            
            # eval
            if (step + 1) % self.eval_steps == 0 and self.with_eval:
                self.eval()
                lr_scheduler.step()
                self.step_list.append(step + 1)
                self.logger.plot_acc(self.step_list, self.mIoU_list, self.mf1_list)
                if self.early_stopping > 10:
                    break
        if not self.with_eval:
            self.logger.save_model(self.model)
        self.logger.log_finish(self.best_iou)

    def eval(self):
        self.model.eval()
        self.model.load_state_dict(torch.load(self.weight, map_location="cpu"))
        self.model.to(self.device)
        self.test_loader = build_dataloader(self.test_pipeline)
        running_metrics_val = runningScore(self.n_classes)
        with torch.no_grad():
            for val_img, val_mask in tqdm(self.test_loader, desc="Valid", ncols=150):
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
                vis_label(pred_list, gt)
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
        self.total_steps = args.total_steps
        self.eval_steps = args.eval_steps
        self.last_step = args.last_step
        self.n_classes = args.n_classes
        self.early_stopping = args.early_stopping
        self.device = args.device
        self.weight = args.weight
        self.with_eval = args.with_eval
        self.log_steps = args.log_steps

        if args.resume:
            self.model.load_state_dict(torch.load(args.resume, map_location="cpu"))
            print(f"Restored from {self.last_step} step!")
            self.logger.info(f"Restored from {self.last_step} step, model:{args.resume}")

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
    R = label.copy()   # 红色通道
    R[R == 1] = 0
    R[R == 2] = 0
    R[R == 3] = 255
    R[R == 4] = 127
    G = label.copy()   # 绿色通道
    G[G == 1] = 0
    G[G == 2] = 255
    G[G == 3] = 0
    R[G == 4] = 127
    B = label.copy()   # 蓝色通道
    B[B == 1] = 255
    B[B == 2] = 0
    B[B == 3] = 0
    R[B == 4] = 127
    return np.dstack((R,G,B))

if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    if args.do_eval:
        trainer.eval()
    else:
        trainer.train()
