import os
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd

# modify here
dataset_path = "/home/xiangjianjian/dataset/gaofen_aug"
target_path = "/home/xiangjianjian/dataset/gaofen_aug"


def convert_255(label_path):
    label = Image.open(label_path)
    label = np.asarray(label)
    label[label > 1] = 1
    return label


if __name__ == "__main__":
    for filename in tqdm(os.listdir(os.path.join(dataset_path, 'gt'))):
        if filename.endswith("_label.png"):
            filepath = os.path.join(dataset_path, 'gt', filename)
            target = os.path.join(dataset_path, 'label', filename.replace("_label.png", ".png"))
            
            label = convert_255(filepath)
            cv2.imwrite(target, label)
