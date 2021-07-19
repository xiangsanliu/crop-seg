# data_url : https://www.kaggle.com/c/carvana-image-masking-challenge/data
from typing import Tuple
import torch
import numpy as np
from PIL import Image
import glob
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


class ADE20KLoader(Dataset):
    def __init__(self, root_path, split='training'):
        super(ADE20KLoader, self).__init__()
        self.img_url = sorted(glob.glob(root_path + '/images/'+split+'/*.jpg'))
        self.mask_url = sorted(glob.glob(root_path + '/annotations/'+split+'/*.png'))


    def __getitem__(self, idx):
        img = Image.open(self.img_url[idx]).convert("RGB")
        img = img.resize((512, 512), Image.BILINEAR)
        img_array = np.array(img, dtype=np.float32) / 255
        mask = Image.open(self.mask_url[idx])
        mask = mask.resize((512, 512), Image.NEAREST)
        mask = np.array(mask, dtype=np.int64)
        img_array = img_array.transpose(2, 0, 1)

        return torch.tensor(img_array.copy()), torch.tensor(mask.copy())

    def __len__(self):
        # return 100
        return len(self.img_url)


if __name__ == "__main__":
    local_path = "/home/xiangjianjian/Projects/spectral-setr/dataset/ADEChallengeData2016"
    dst = ADE20KLoader(local_path, split='training')
    trainloader = DataLoader(dst, batch_size=4)
    for i, data_samples in enumerate(trainloader):
        imgs, labels = data_samples
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
            for j in range(4):
                plt.imshow(dst.decode_segmap(labels.numpy()[j]))
                plt.show()
