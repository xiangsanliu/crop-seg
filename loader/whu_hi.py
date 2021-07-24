import os
import glob
import torch
from scipy.io import loadmat
import numpy as np

from torch.utils.data import Dataset

class SpectralDataset(Dataset):
    def __init__(self, root_path, data_type):
        super(SpectralDataset, self).__init__()
        self.img_path = sorted(glob.glob(os.path.join(root_path, data_type, '*.mat')))
    def __getitem__(self, index):
        mat = loadmat(self.img_path[index])
        img = mat['img']
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
    dataset = SpectralDataset(root_path, data_type)
    img, mask = dataset.__getitem__(0)
    print(img.shape)
    print(mask.shape)