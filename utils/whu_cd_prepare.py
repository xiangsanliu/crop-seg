import os
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd

def convert_255(label_path):
    label = Image.open(label_path)
    label = np.asarray(label)
    label[label > 1] = 1
    return label

def convert_label():
    root_dir = 'dataset/whu-cd'
    
    label_path = os.path.join(root_dir, 'label')
    image_path = os.path.join(root_dir, 'image')
    target_path = os.path.join(root_dir, 'label_converted')
    
    for image_name in os.listdir(image_path):
        label = convert_255(os.path.join(label_path, image_name))
        cv2.imwrite(os.path.join(target_path, image_name), label)
        
        
if __name__ == "__main__":
    convert_label()
    