import glob
import torch
from scipy.io import loadmat
import numpy as np

from torch.utils.data import Dataset

class SpectralDataset(Dataset):
    def __init__(self):
        super(SpectralDataset, self).__init__()
        self.img_path = sorted(glob.glob('data/WHU-Hi/spectral/HongHu_*.mat'))
    def __getitem__(self, index):
        mat = loadmat(self.img_path[index])
        img = mat['img']
        img_gt = mat['img_gt'].astype(np.int64)
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        img_gt = torch.from_numpy(img_gt)
        return img, img_gt
    def __len__(self):
        return len(self.img_path)