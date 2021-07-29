import os
import glob
import torch
from scipy.io import loadmat
import numpy as np
from tools.data import load_data, random_patch

from torch.utils.data import Dataset


class SpectralDataset(Dataset):
    def __init__(self, patch_path, data_type, img_size,
                 mode='train',
                 original_data_path='/home/xiangjianjian/dataset/WHU-Hi/',
                 all_random=False):
        super(SpectralDataset, self).__init__()
        self.mode = mode
        self.img_size = img_size
        self.all_random = all_random
        parent_path = os.path.join(patch_path, data_type + str(img_size))
        if all_random:
            parent_path = parent_path + 'random'
        if self.mode == 'test':
            self.img, self.gt = load_data(original_data_path, data_type)
        if self.mode == 'train' or self.mode == 'val':
            self.img_path = sorted(
                glob.glob(os.path.join(parent_path, self.mode+'*.mat')))

    def __getitem__(self, index):
        if self.mode == 'test':
            patch_img, patch_gt = random_patch(
                self.img, self.gt, self.img_size)
            patch_img = np.transpose(patch_img, (2, 0, 1))
            return torch.from_numpy(patch_img), torch.from_numpy(patch_gt)
        else:
            mat = loadmat(self.img_path[index])
            img = mat['img'].astype(np.float32)
            gt = mat['gt'].astype(np.int64)
            img = torch.from_numpy(img)
            # img = img.permute(2, 0, 1)
            gt = torch.from_numpy(gt)
            return img, gt

    def __len__(self):
        if self.mode == 'test':
            return 100
        else:
            return len(self.img_path)


if __name__ == '__main__':
    patch_path = '/home/xiangjianjian/Projects/spectral-setr/data/WHU-Hi/patch'
    data_type = 'WHU_Hi_HanChuan'
    img_size = 224
    dataset = SpectralDataset(root_path, data_type, img_size)
    for i in range(len(dataset)):
        img, mask = dataset.__getitem__(i)
        # print(img.shape)
        print(torch.max(mask))
