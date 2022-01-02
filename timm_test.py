import torch
import torch.nn as nn
from tqdm import tqdm

from model import build_model
import configs


config = getattr(configs, 'indicesformer_b4_tianchi')
# config = getattr(configs, 'unet_tianchi2_label_no_overlap')
model = build_model(config["model"])
x = torch.randn(1, 7, 512, 512)
x = model(x)
print(x.shape)

# criterion = RMILoss(num_classes=2, ignore_index=255)

# net = HRNet_Mscale(num_classes=2, criterion=criterion)

# net.load_ckpt("pretrained/hrnet.pth")

# x = torch.randn(2, 3, 512, 512)

# net.eval()
# inputs = {"images": x}

# x = net(inputs)

# for k, v in x.items():
#     print(k, v.shape)
