"""
Author      : now more
Connect     : lin.honghui@qq.com
LastEditors: Please set LastEditors
Description : 
LastEditTime: 2020-11-28 08:37:52
"""

from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
import os


class PNG_Dataset(Dataset):

    def __init__(self, csv_file, image_dir, mask_dir, transforms=None):
        """
        Description: 
        Args (type): 
            csv_file  (string): Path to the file with annotations, see `utils/data_prepare` for more information.
            image_dir (string): Derectory with all images.
            mask_dir (string): Derectory with all labels.
            transforms (callable,optional): Optional transforms to be applied on a sample.
        return: 
        """
        self.csv_file = pd.read_csv(csv_file, header=None)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): index of sample
        """
        filename = self.csv_file.iloc[idx, 0]
        _, filename = os.path.split(filename)
        image_path = os.path.join(self.image_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        sample = {"image": image, "mask": mask}

        if self.transforms:
            sample = self.transforms(sample)

        image, mask = sample["image"], sample["mask"]

        return image, mask.long()


class Inference_Dataset(Dataset):
    def __init__(self, image_dir, csv_file, transforms=None):
        """
        Description: 
        Args (type): 
            csv_file  (string): Path to the file with annotations, see `utils/data_prepare` for more information.
            image_dir (string): Derectory with all images.
            transforms (callable,optional): Optional transforms to be applied on a sample.
        return: 
        """
        self.image_dir = image_dir
        self.csv_file = pd.read_csv(csv_file, header=None)
        self.transforms = transforms

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        filename = self.csv_file.iloc[idx, 0]
        _, filename = os.path.split(filename)
        image_path = os.path.join(self.image_dir, filename)
        image = np.asarray(Image.open(image_path).convert("RGB"))  # mode:RGBA

        sample = {"image": image}
        if self.transforms:
            sample = self.transforms(sample)
        image = sample["image"]

        pos_list = self.csv_file.iloc[idx, 1:].values.astype(
            "int"
        )  # ---> (topleft_x,topleft_y,buttomright_x,buttomright_y)
        return image, pos_list


class ConcatDataset(Dataset):
    def __init__(
        self,
        csv_file1,
        csv_file2,
        image_dir1,
        image_dir2,
        label_dir1,
        label_dir2,
        transforms=None,
    ):

        self.filenames = self._concat(
            csv_file1, csv_file2, image_dir1, image_dir2, label_dir1, label_dir2
        )
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image_path, mask_path = self.filenames[idx]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        sample = {"image": image, "mask": mask}

        if self.transforms:
            sample = self.transforms(sample)

        image, mask = sample["image"], sample["mask"]

        return image, mask.long()

    def _concat(
        self, csv_file1, csv_file2, image_dir1, image_dir2, label_dir1, label_dir2,
    ):
        self.csv_file1 = pd.read_csv(csv_file1)
        self.csv_file2 = pd.read_csv(csv_file2)
        filenames = []
        for i in range(len(csv_file1)):
            filename = self.csv_file1.iloc[i, 0]
            _, filename = os.path.split(filename)
            image_path = os.path.join(image_dir1, filename)
            label_path = os.path.join(label_dir1, filename)
            filenames.append((image_path, label_path))
        for i in range(len(csv_file2)):
            filename = self.csv_file2.iloc[i, 0]
            _, filename = os.path.split(filename)
            image_path = os.path.join(image_dir2, filename)
            label_path = os.path.join(label_dir2, filename)
            filenames.append((image_path, label_path))
        return filenames

