import os
import math
import torch
import numpy as np
from utils.data import load_data, random_patch, random_crop, sorted_patch, patch
from utils.utils import standartize
from spectral_seg import get_predict
from matplotlib import pyplot as plt

# 定义全局变量
ROOT_PATH = '/home/xiangjianjian/dataset/WHU-Hi/'
DATA_TYPE = 'WHU_Hi_HanChuan'


def main():
    img_size = 256
    new_data_path = os.path.join(
        ROOT_PATH, 'patch', DATA_TYPE + str(img_size))  # 存放数据路径 patch是文件夹名称
    print(new_data_path)
    img, gt = load_data(ROOT_PATH, DATA_TYPE)

    # crop_img(img, gt, DATA_TYPE, new_data_path, img_size=img_size)
    random_crop(img, gt, DATA_TYPE, new_data_path +
                'random', img_size=img_size)


def split_predict():
    img, gt = load_data(ROOT_PATH, DATA_TYPE)
    print(img.shape)
    img_size = 256
    c, h, w = img.shape
    num_x = math.ceil(w / img_size)
    num_y = math.ceil(h / img_size)
    pred_all = np.zeros((num_y * img_size, num_x*img_size))
    print('pred_all', pred_all.shape)
    img_data = []
    patch_edge = []
    for j in range(num_y):
        for i in range(num_x):
            a = i * img_size
            b = j * img_size
            if a+img_size > w:
                a = a - (a+img_size-w)
            if b+img_size > h:
                b = b - (b+img_size-h)
            patch_edge.append([a, b])
            patched_img, patched_gt = patch(img, gt, img_size, a, b)
            patched_img = np.transpose(patched_img, (2, 0, 1))
            img_data.append(standartize(patched_img))
    img_data = np.array(img_data, dtype=np.float32)
    img_data = torch.from_numpy(img_data)
    pred = get_predict(img_data)
    print('pred', pred.shape)
   
    for i in range(num_y*num_x):
        a = patch_edge[i][0]
        b = patch_edge[i][1]
        a1 = a+img_size
        b1 = b+img_size
        pred_all[b:b1, a:a1] = pred[i]
    
    pred_all = pred_all[0:h, 0:w]
    print(pred_all.shape)
    plt.imshow(pred_all, cmap="tab20c")
    plt.rcParams['savefig.dpi'] = 500
    plt.axis('off')
    plt.savefig('./work/predict_all.jpg', bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    split_predict()
    # main()
    pass
