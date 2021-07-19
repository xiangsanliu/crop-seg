# data_url : https://www.kaggle.com/c/carvana-image-masking-challenge/data
from typing import Tuple
import torch
import numpy as np
from setr.SETR import SETRModel
from setr.trans_config import TransConfig
from loader.ade20k_loader import ADE20KLoader
from loader.ade import ADE20KSegmentation
from tools.metrics import runningScore
from PIL import Image
import glob
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F




def build_model():
    config = TransConfig(
        patch_size=(32, 32),
        in_channels=3,
        out_channels=1,
        embed_dim=768,
        num_hidden_layers=24,
        num_heads=8,
        sample_rate=4,
        num_classes=151
    )

    model = SETRModel(config)
    return model





def compute_dice(input, target):
    eps = 0.0001
    # input 是经过了sigmoid 之后的输出。
    input = (input > 0.5).float()
    target = (target > 0.5).float()

    # inter = torch.dot(input.view(-1), target.view(-1)) + eps
    inter = torch.sum(target.view(-1) * input.view(-1)) + eps

    # print(self.inter)
    union = torch.sum(input) + torch.sum(target) + eps

    t = (2 * inter.float()) / union.float()
    return t


def predict():
    model = build_model()
    model.load_state_dict(torch.load(
        "./checkpoints/SETR_ade.pkl", map_location="cpu"))
    # print(model)

    root_path = "/home/xiangjianjian/Projects/spectral-setr/dataset/ADEChallengeData2016"
    val_set = ADE20KLoader(root_path, split='validation')


    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    with torch.no_grad():
        for img, mask in val_loader:
            pred = model(img)
            pred = pred.data.max(1)[1].cpu().numpy()
            plt.subplot(1, 3, 1)
            img = img.permute(0, 2, 3, 1)
            plt.imshow(img[0])
            plt.subplot(1, 3, 2)
            plt.imshow(pred[0], cmap="gray")
            plt.subplot(1, 3, 3)
            plt.imshow(mask[0], cmap="gray")
            plt.savefig('./predict.jpg')
            plt.show()


def valid(model, val_loader):
    model.eval()
    n = 0
    # dice = 0.0
    val_loss_sum = 0.0
    with torch.no_grad():
        for val_img, val_mask in tqdm(val_loader, total=len(val_loader), desc='Valid', unit=' step'):
            n += 1
            val_img = val_img.to(device)
            val_mask = val_mask.to(device)
            pred_img = model(val_img)
            val_loss = loss_func(pred_img, val_mask)
            pred = pred_img.data.max(1)[1].cpu().numpy()
            gt = val_mask.data.cpu().numpy()
            running_metrics_val.update(gt, pred)
            val_loss_sum += val_loss.item()
            # pred_img = torch.sigmoid(model(val_img))
            # if out_channels == 1:
            #     pred_img = pred_img.squeeze(1)
            # cur_dice = compute_dice(pred_img, val_mask)
            # dice += cur_dice
        # dice = dice / n
        score, class_iou = running_metrics_val.get_scores()
        print("val loss is ", val_loss_sum/n)
        for k, v in score.items():
            print(k, v)
            # logger.info("{}: {}".format(k, v))
            # writer.add_scalar("val_metrics/{}".format(k), v, i + 1)

        # for k, v in class_iou.items():
            # print(k, v)
            # logger.info("{}: {}".format(k, v))
            # writer.add_scalar("val_metrics/cls_{}".format(k), v, i + 1)

        running_metrics_val.reset()
        torch.save(model.state_dict(),
                   "./checkpoints/SETR_ade.pkl")
        model.train()


def train():
    step = 0
    report_loss = 0.0
    loss_list = []
    epoch_list = []

    for epoch in range(epoches):

        for img, mask in tqdm(train_loader, total=len(train_loader), desc=f"Train {epoch+1}/{epoches}", unit=' step'):

            optimizer.zero_grad()
            step += 1
            img = img.to(device)
            mask = mask.to(device)
            pred_img = model(img)  # pred_img (batch, len, channel, W, H)
            # if out_channels == 1:
            #     pred_img = pred_img.squeeze(1)  # 去掉通道维度

            loss = loss_func(pred_img, mask)
            report_loss += loss.item()
            loss.backward()
            optimizer.step()
        print('mean loss:', report_loss / step)
        loss_list.append(report_loss / step)
        epoch_list.append(epoch)
        valid(model, val_loader)
        step = 0
        report_loss = 0.0
    plt.plot(epoch_list, loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('./result.jpg')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epoches = 30
out_channels = 1

model = build_model()
model.to(device)
batch_size = 16
num_workers = 8
running_metrics_val = runningScore(151)

root_path = "/home/xiangjianjian/Projects/spectral-setr/dataset/ADEChallengeData2016"
train_set = ADE20KLoader(root_path, split='training')
val_set = ADE20KLoader(root_path, split='validation')


train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=True,
                        shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=batch_size, drop_last=True,
                        shuffle=False, num_workers=num_workers, pin_memory=True)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=1e-5, weight_decay=1e-4)


train()
# predict()
