'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
LastEditTime: 2020-11-28 04:04:40
Description : 
'''
import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser(description="")
    parser.add_argument("-root_dir",type=str)
    arg = parser.parse_args()
    root_dir = arg.root_dir
    save_dir = root_dir

    image_1_csv = pd.read_csv(os.path.join(root_dir,'image_1.csv'),header=None)
    image_2_csv = pd.read_csv(os.path.join(root_dir,'image_2.csv'),header=None)

    total_csv = pd.concat((image_1_csv, image_2_csv),axis=0)

    total_csv.to_csv(os.path.join(save_dir,"train.csv"),header=None,index=None)