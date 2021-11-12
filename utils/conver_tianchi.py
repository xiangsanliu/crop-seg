import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import random

data_root = "dataset/tianchi/round2_no_overlap"

image_dir = os.path.join(data_root, "image")
label_dir = os.path.join(data_root, "label")
target_dir = os.path.join(data_root, "label_vis")
target_dir_val = os.path.join(target_dir, "val")
target_dir_train = os.path.join(target_dir, "train")

os.makedirs(target_dir, exist_ok=True)
os.makedirs(target_dir_val, exist_ok=True)
os.makedirs(target_dir_train, exist_ok=True)

all_files = os.listdir(image_dir)

def convert_label(label):
    label = np.asarray(label)
    R = label.copy()   # 红色通道
    R[R == 1] = 0
    R[R == 2] = 0
    R[R == 3] = 255
    R[R == 4] = 127
    G = label.copy()   # 绿色通道
    G[G == 1] = 0
    G[G == 2] = 255
    G[G == 3] = 0
    R[G == 4] = 127
    B = label.copy()   # 蓝色通道
    B[B == 1] = 255
    B[B == 2] = 0
    B[B == 3] = 0
    R[B == 4] = 127
    return np.dstack((R,G,B))

def convert_data(file_names):
    for name in tqdm(file_names):
        image_path = os.path.join(image_dir, name)
        label_path = os.path.join(label_dir, name)
        image = Image.open(image_path).convert("RGB")
        image = np.asarray(image)
        label = Image.open(label_path)
        label = convert_label(label)
        target = np.concatenate((image, label), axis = 1)
        if name.startswith("image_11"):
            target_path = os.path.join(target_dir_val, name)
        else:
            target_path = os.path.join(target_dir_train, name)
        
        cv2.imwrite(target_path, target)
        
if __name__ == '__main__':
    convert_data(all_files)