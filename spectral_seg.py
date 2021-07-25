import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torch.nn.functional as F
import time

from model.unet_model import UNet
from model.setr.SETR import SETR_MLA
from loader.whu_hi import SpectralDataset
from tools.metrics import runningScore

train_config = dict(
    batch_size=16,
    num_workers=4,
    data_path='/home/xiangjianjian/Projects/spectral-setr/data/WHU-Hi/patch',
    data_type='WHU_Hi_HanChuan',
    data_img_size=224,
    n_classes=17,
    in_channels=274,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    lr=1e-5,
    train_val_rate=0.9,
    pth_path='checkpoints/SETR_MLA.pkl'
)

# ============= Model Prepare =============
# model = UNet(n_channels=train_config['in_channels'],
#              n_classes=train_config['n_classes'])

model = SETR_MLA(
    img_dim=224,
    patch_dim=16,
    num_channels=274,
    num_classes=17,
    embedding_dim=1024,
    num_heads=16,
    num_layers=24,
    hidden_dim=4096,
    dropout_rate=0.1,
    attn_dropout_rate=0.1
)

# ============= Data Prepare =============
dataset = SpectralDataset(patch_path=train_config['data_path'],
                          data_type=train_config['data_type'],
                          img_size=train_config['data_img_size'],
                          all_random=True)
print(dataset.__len__())
trainlen = int(train_config['train_val_rate']*len(dataset))
lengths = [trainlen, len(dataset)-trainlen]
train_set, val_set = random_split(dataset, lengths)
# train_set = SpectralDataset(patch_path=train_config['data_path'],
#                           data_type=train_config['data_type'],
#                           img_size=train_config['data_img_size'],
#                           mode='train',
#                           all_random=True)
# val_set = SpectralDataset(patch_path=train_config['data_path'],
#                           data_type=train_config['data_type'],
#                           img_size=train_config['data_img_size'],
#                           mode='val',
#                           all_random=True)
train_loader = DataLoader(train_set,
                          batch_size=train_config['batch_size'],
                          drop_last=False,
                          shuffle=True,
                          num_workers=train_config['num_workers'],
                          pin_memory=True)
val_loader = DataLoader(val_set,
                        batch_size=train_config['batch_size'],
                        drop_last=False,
                        shuffle=False,
                        num_workers=train_config['num_workers'],
                        pin_memory=True)

# ============= Train Settings =============
running_metrics_val = runningScore(train_config['n_classes'])
loss_func = nn.CrossEntropyLoss()
device = train_config['device']
BEST_IOU = 0.
BEST_SCORE = {}


def predict():
    test_dataset = SpectralDataset(patch_path=train_config['data_path'],
                                   data_type=train_config['data_type'],
                                   img_size=train_config['data_img_size'],
                                   mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model.load_state_dict(torch.load(
        train_config['pth_path'], map_location='cpu'))

    with torch.no_grad():
        for img, mask in test_loader:
            pred = model(img)
            pred = pred.data.max(1)[1].cpu().numpy()
            plt.subplot(1, 3, 1)
            img = img.permute(0, 2, 3, 1)
            img = img[0]
            plt.imshow(img[:, :, 0:3])
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
        score, class_iou, mean_iu = running_metrics_val.get_scores()
        print("val loss:\t", val_loss_sum / n)
        for k, v in score.items():
            print(k, v)

        running_metrics_val.reset()
        global BEST_IOU
        global BEST_SCORE
        if mean_iu > BEST_IOU:
            BEST_SCORE = score
            BEST_IOU = mean_iu
            print('Saving Best Model...')
            torch.save(model.state_dict(), train_config['pth_path'])
        model.train()


def train():
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_config['lr'], weight_decay=1e-5)
    step = 0
    report_loss = 0.0
    loss_list = []
    epoch_list = []
    epoches = 100
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
        print('train loss:', report_loss / step)
        loss_list.append(report_loss / step)
        epoch_list.append(epoch)
        valid(model, val_loader)
        step = 0
        report_loss = 0.0
    print('Best score:')
    for k, v in BEST_SCORE.items():
        print(k, v)
    plt.plot(epoch_list, loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('./work/loss.jpg')


if __name__ == '__main__':
    train()
    # predict()
    pass
