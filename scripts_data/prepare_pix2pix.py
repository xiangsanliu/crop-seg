from enum import unique
import os
from re import L
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import random
import shutil

def judge_label(label):
    num, count = np.unique(label, return_counts=True)
    if len(num) == 1:
        return False
    if count[0] > 173015 and num[0] == 0:
        return False
    return True

source = "tianchi_no"
target = "tianchi_pix_1"

os.makedirs(target, exist_ok=True)

train_img_path = os.path.join(target, "train_img")
train_label_path = os.path.join(target, "train_label")
val_img_path = os.path.join(target, "val_img")
val_label_path = os.path.join(target, "val_label")
os.makedirs(train_img_path, exist_ok=True)
os.makedirs(train_label_path, exist_ok=True)
os.makedirs(val_img_path, exist_ok=True)
os.makedirs(val_label_path, exist_ok=True)

for i in tqdm(os.listdir(os.path.join(source, "image"))):
    source_img = os.path.join(source, "image", i)
    source_label = os.path.join(source, "label", i)
    label = Image.open(source_label)
    if not judge_label(label):
        continue
    if i.startswith("image_11"):
        target_img = os.path.join(val_img_path, i)
        target_label = os.path.join(val_label_path, i)
    else:
        target_img = os.path.join(train_img_path, i)
        target_label = os.path.join(train_label_path, i)
    shutil.copy(source_img, target_img)
    shutil.copy(source_label, target_label)
