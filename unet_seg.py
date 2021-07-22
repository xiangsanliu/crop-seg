import torch
import glob
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torch.nn.functional as F
import time

from model.unet_model import UNet
from loader.whu_hi import SpectralDataset
from tools.metrics import runningScore

running_metrics_val = runningScore(23)
loss_func = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def predict():
    model = UNet(n_channels=270, n_classes=23)
    model.load_state_dict(torch.load('checkpoints/unet.pkl', map_location='cpu'))
    dataset = SpectralDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for img, mask in dataloader:
            pred = model(img)
            pred = pred.data.max(1)[1].cpu().numpy()
            plt.subplot(1, 3, 1)
            img = img.permute(0, 2, 3, 1)
            img = img[0]
            plt.imshow(img[:,:,0:3])
            plt.subplot(1, 3, 2)
            plt.imshow(pred[0], cmap="gray")
            plt.subplot(1, 3, 3)
            plt.imshow(mask[0], cmap="gray")
            plt.savefig('./work/predict.jpg')
            plt.show()
            time.sleep(2)
 

def valid(model, val_loader):
    model.eval()
    n = 0
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
        score, class_iou = running_metrics_val.get_scores()
        print("val loss is ", val_loss_sum/n)
        for k, v in score.items():
            print(k, v)

        running_metrics_val.reset()
        torch.save(model.state_dict(),
                   "./checkpoints/unet.pkl")
        model.train()

def train():
    dataset = SpectralDataset()
    model_config = {
        'img_channels': 270,
        'output_channels': 9
    }
    batch_size = 4
    num_workers = 8
    # model = UNet(config=model_config)
    model = UNet(n_channels=270, n_classes=23)
    trainlen = int(0.9*len(dataset))
    lengths = [trainlen, len(dataset)-trainlen]
    train_set, val_set = random_split(dataset, lengths)
    train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=True,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, drop_last=True,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, weight_decay=1e-5)

    step = 0
    report_loss = 0.0
    loss_list = []
    epoch_list = []
    epoches = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(epoches):

        for img, mask in tqdm(train_loader, total=len(train_loader), desc=f"Train: {epoch+1}/{epoches}", unit=' step'):
            optimizer.zero_grad()
            step += 1
            img = img.to(device)
            mask = mask.to(device)
            pred_img = model(img)  # pred_img (batch, len, channel, W, H)
            # if out_channels == 1:
                # pred_img = pred_img.squeeze(1)  # 去掉通道维度

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
    plt.savefig('./work/loss.jpg')

if __name__== '__main__':
    # train()
    predict()