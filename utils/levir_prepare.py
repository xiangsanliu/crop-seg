import os
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd

root_path = 'dataset/levir'

image_save_path = os.path.join(root_path, 'image')
gt_save_path = os.path.join(root_path, 'gt')

a_dir = os.path.join(root_path, 'A')
b_dir = os.path.join(root_path, 'B')

for name in os.listdir(a_dir):
    
    
    basename = name.split('.')[0]
    print(basename)