import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt


from utils.model_tools import ModelValidator
from data.dataloader import build_dataloader
from model import build_model
from configs.segformer_tianchi import config

model = build_model(config['model'])
train_loader, val_loader = build_dataloader(config['train_pipeline'])
train_config = config['train_config']
lr_scheduler_config = config['lr_scheduler']



# ============= Data Prepare =============
# train_set = SpectralDataset(patch_path=train_config['data_path'],
#                             data_type=train_config['data_type'],
#                             img_size=train_config['data_img_size'],
#                             mode='train',
#                             all_random=True)
# val_set = SpectralDataset(patch_path=train_config['data_path'],
#                           data_type=train_config['data_type'],
#                           img_size=train_config['data_img_size'],
#                           mode='val',
#                           all_random=True)

# train_loader = DataLoader(train_set,
#                           batch_size=train_config['batch_size'],
#                           drop_last=False,
#                           shuffle=True,
#                           num_workers=train_config['num_workers'],
#                           pin_memory=True)
# val_loader = DataLoader(val_set,
#                         batch_size=train_config['batch_size'],
#                         drop_last=False,
#                         shuffle=False,
#                         num_workers=train_config['num_workers'],
#                         pin_memory=True)


loss_func = nn.CrossEntropyLoss()
device = train_config['device']


validator = ModelValidator(train_config, loss_func)


def predict():
    model.load_state_dict(torch.load(
        f"checkpoints/{train_config['model_type']}.pkl", map_location='cpu'))
    # validator.validate_model(model, val_loader, 'cpu')
    test_set = SpectralDataset(patch_path=train_config['data_path'],
                               data_type=train_config['data_type'],
                               img_size=train_config['data_img_size'],
                               mode='test')
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    validator.predict(model, test_loader)
 

def get_predict(img_data):
    model.load_state_dict(torch.load(
        f"checkpoints/{train_config['model_type']}.pkl", map_location='cpu'))
    pred = validator.draw_predict(model, img_data)
    return pred

def train():
    if (train_config['restore']):
        model.load_state_dict(torch.load(
            f"checkpoints/{train_config['model_type']}.pkl", map_location='cpu'))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_config['lr'], weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **lr_scheduler_config)
    step = 0
    report_loss = 0.0
    loss_list = []
    epoch_list = []
    model.to(device)
    for epoch in range(train_config['epoches']):

        for img, mask in tqdm(train_loader, total=len(train_loader), desc=f"Train: {epoch+1}/{train_config['epoches']},lr={lr_scheduler.get_lr()}", unit=' step'):
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
        print('Train loss:\t', report_loss / step)
        loss_list.append(report_loss / step)
        epoch_list.append(epoch)
        # valid(model, val_loader)
        best_score = validator.validate_model(model, val_loader, device)
        step = 0
        report_loss = 0.0
        lr_scheduler.step()
    print('\nTraining process finished.')
    print('Best score:')
    for k, v in best_score.items():
        print(k, v)
    plt.plot(epoch_list, loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(f"./work/{train_config['model_type']}.jpg")


if __name__ == '__main__':
    if train_config['mode'] == 'train':
        train()
    else:
        predict()
    # print(model)
    # print(train_loader)
    pass
