import os
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd

# modify here
dataset_path = "/home/xiangjianjian/dataset/gaofen"
target_path = "/home/xiangjianjian/dataset/gaofen"


def convert_255(label_path):
    label = Image.open(label_path)
    label = np.asarray(label)
    label[label > 1] = 1
    return label


def convert_gt():
    gt_path = os.path.join(dataset_path, "gt")
    gt_path_target = os.path.join(target_path, "label")

    for i in tqdm(range(2000)):
        label_a = os.path.join(gt_path, f"{i+1}_1_label.png")
        label_b = os.path.join(gt_path, f"{i+1}_2_label.png")
        label_a_target = os.path.join(gt_path_target, f"{i+1}_1.png")
        label_b_target = os.path.join(gt_path_target, f"{i+1}_2.png")

        label_a = convert_255(label_a)
        label_b = convert_255(label_b)
        # a = np.sum(label_a > 1)
        # if a > 0:
        #     print(i+1, a)
        cv2.imwrite(label_a_target, label_a)
        cv2.imwrite(label_b_target, label_b)


def split_dataset(ratio=0.8):
    file_names = []
    for i in range(2000):
        file_names.append(f"{i+1}_1.png")
        file_names.append(f"{i+1}_2.png")
    file_names = np.array(file_names)
    np.random.shuffle(file_names)
    split_index = int(ratio * len(file_names))
    train_names = file_names[:split_index]
    val_names = file_names[split_index:]
    write_list(train_names, split="train")
    write_list(val_names, split="val")


def write_list(names, split="train"):
    names = pd.DataFrame(names)
    names.to_csv(os.path.join(target_path, f"{split}.csv"), header=None, index=None)


if __name__ == "__main__":
    convert_gt()
    # split_dataset(ratio=0.9)
    