import os
import numpy as np
import pandas as pd

target_path = "dataset/gaofen_aug"


def split_seg(root_dir, ratio=0.9):
    gaofen = []
    whu_cd = []
    for i in os.listdir(root_dir):
        if "t" in i:
            whu_cd.append(i)
        else:
            gaofen.append(i)
    gaofen = np.asarray(gaofen)
    split_index = int(ratio * len(gaofen))
    train_names = gaofen[:split_index]
    val_names = gaofen[split_index:]
    train_names = np.append(train_names, np.asarray(whu_cd))
    write_list(train_names, type="seg", split="train")
    write_list(val_names,type="seg", split="val")


def split_dataset(root_dir, ratio=0.9):
    filenames = set()
    gaofen = set()
    whu_cd = set()
    for i in os.listdir(root_dir):
        if i.endswith("_1.png"):
            filenames.add(i.replace("_1.png", ""))
    for i in filenames:
        if "t" in i:
            whu_cd.add(i)
        else:
            gaofen.add(i)

    gaofen = np.array(list(gaofen))
    split_index = int(ratio * len(gaofen))
    train_names = gaofen[:split_index]
    train_names = np.append(train_names, np.asarray(list(whu_cd)))
    val_names = gaofen[split_index:]
    write_list(train_names,type="cd", split="train")
    write_list(val_names, type="cd", split="val")


def write_list(names,type, split="train"):
    names = pd.DataFrame(names)
    names.to_csv(os.path.join(target_path, f"{type}_{split}.csv"), header=None, index=None)


if __name__ == "__main__":
    root_dir = f"{target_path}/image"
    split_dataset(root_dir)
    split_seg(root_dir, ratio=0.95)
