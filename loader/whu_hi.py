import os
import glob
import torch
from scipy.io import loadmat
import numpy as np

from torch.utils.data import Dataset

class SpectralDataset(Dataset):
    def __init__(self, root_path, data_type, img_size):
        super(SpectralDataset, self).__init__()
        self.img_path = sorted(glob.glob(os.path.join(root_path, data_type + str(img_size), '*.mat')))
    def __getitem__(self, index):
        mat = loadmat(self.img_path[index])
        img = mat['img'].astype(np.float32)
        gt = mat['gt'].astype(np.int64)
        img = torch.from_numpy(img)
        # img = img.permute(2, 0, 1)
        gt = torch.from_numpy(gt)
        return img, gt
    def __len__(self):
        return len(self.img_path)

if __name__ == '__main__':
    root_path = '/home/xiangjianjian/Projects/spectral-setr/data/WHU-Hi/patch'
    data_type = 'WHU_Hi_HanChuan'
    img_size = 224
    dataset = SpectralDataset(root_path, data_type, img_size)
    for i in range(len(dataset)):
        img, mask = dataset.__getitem__(i)
        # print(img.shape)
        print(torch.max(mask))