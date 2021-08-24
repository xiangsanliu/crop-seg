"""
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
LastEditTime: 2020-11-28 04:04:40
Description : 
"""
import os
import pandas as pd
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser(description="")
    parser.add_argument("-root_dir", type=str)
    arg = parser.parse_args()
    root_dir = arg.root_dir
    save_dir = root_dir

    image_10_csv = pd.read_csv(os.path.join(root_dir, "image_10.csv"), header=None)
    image_11_csv = pd.read_csv(os.path.join(root_dir, "image_11.csv"), header=None)
    image_20_csv = pd.read_csv(os.path.join(root_dir, "image_20.csv"), header=None)
    image_21_csv = pd.read_csv(os.path.join(root_dir, "image_21.csv"), header=None)

    total_csv = pd.concat(
        (image_10_csv, image_11_csv, image_20_csv, image_21_csv), axis=0
    )

    data: pd.DataFrame = total_csv.sample(frac=1.0)
    rows, cols = data.shape
    split_index = int(rows * 0.8)
    train_csv: pd.DataFrame = data.iloc[0:split_index, :]
    test_csv: pd.DataFrame = data.iloc[split_index:, :]

    train_csv.to_csv(os.path.join(save_dir, "train_random.csv"), header=None, index=None)
    test_csv.to_csv(os.path.join(save_dir, "test_random.csv"), header=None, index=None)
