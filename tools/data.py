# -*- coding: utf-8 -*-
"""
Created on Wed May 15 17:17:30 2019

@author: viryl
"""
import os
import math
import scipy.io
import scipy.ndimage
import numpy as np
from random import shuffle
from utils import pca, pad, standartize, patch
from scipy.io import loadmat, savemat


# 定义全局变量
PATCH_SIZE = 17  # 切片尺寸
OUTPUT_CLASSES = 16  # 输出9类地物
TEST_FRAC = 0.50  # 用来测试数据的百分比
ROOT_PATH = '/home/xiangjianjian/dataset/WHU-Hi/'
DATA_TYPE = 'WHU_Hi_HanChuan'



# 加载数据

def load_data(root_path, data_type):
    img_path = os.path.join(
        root_path, data_type.replace('_', '-'), data_type + '.mat')
    gt_path = os.path.join(root_path, data_type.replace(
        '_', '-'), data_type + '_gt.mat')
    img = loadmat(img_path)
    gt = loadmat(gt_path)
    img = np.array(img[data_type])
    img = np.transpose(img, (2, 0, 1))
    gt = np.array(gt[data_type + '_gt'])
    return img, gt


def patch(img, gt, img_size, a, b):
    patch_img = img[:, b:b+img_size, a:a+img_size]
    patch_gt = gt[b:b+img_size, a:a+img_size]
    c, h, w = patch_img.shape
    new_img = np.zeros((c, img_size, img_size))
    new_gt = np.zeros((img_size, img_size))
    new_img[:, 0:h, 0:w] = patch_img
    new_gt[0:h, 0:w] = patch_gt
    patch_img = new_img
    patch_gt = new_gt
    for i in range(c):
        mean = np.mean(patch_img[i, :, :])
        patch_img[i] = patch_img[i] - mean
    patch_img = np.transpose(patch_img, (1, 2, 0))
    return patch_img, patch_gt


def expand_data(img):
    expanded = []
    expanded.append(img)
    img_90 = np.rot90(img)
    expanded.append(img_90)
    img_180 = np.rot90(img_90)
    expanded.append(img_180)
    img_270 = np.rot90(img_180)
    expanded.append(img_270)
    img_lr = np.fliplr(img)
    expanded.append(img_lr)
    img_ud = np.flipud(img)
    expanded.append(img_ud)
    img_90_lr = np.fliplr(img_90)
    expanded.append(img_90_lr)
    img_90_ud = np.flipud(img_90)
    expanded.append(img_90_ud)
    return expanded


def crop_img(img, gt, type, save_path, img_size=224):
    h, w = gt.shape
    num_x = math.ceil(w / img_size)
    num_y = math.ceil(h / img_size)
    overlap_x = int((num_x * img_size - w) / (num_x - 1))
    jump_x = img_size - overlap_x
    overlap_y = int((num_y * img_size - h) / (num_y - 1))
    jump_y = img_size - overlap_y
    img_result = []
    gt_result = []
    for j in range(num_y):
        for i in range(num_x):
            # a = (i * img_size) if (i * img_size - overlap_x)<0 else (i * img_size - overlap_x)
            # b = (j * img_size) if (j * img_size - overlap_y)<0 else (j * img_size - overlap_y)
            a = i * jump_x
            b = j * jump_y
            patched_img, patched_gt = patch(img, gt, img_size, a, b)
            print(patched_img.shape, patched_gt.shape)
            img_result.append(patched_img)
            gt_result.append(patched_gt)
    num = 1
    for i in range(len(gt_result)):
        expended_img = expand_data(img_result[i])
        expended_gt = expand_data(gt_result[i])
        for j in range(len(expended_img)):
            # h, w = expended_gt[j].shape
            # if h!=img_size or w!=img_size:
            # break

            mat = {
                'img': np.transpose(expended_img[j], (2, 0, 1)),
                'gt': expended_gt[j]
            }
            file_path = os.path.join(save_path, '%s_%02d.mat' % (type, num))
            savemat(file_path, mat)
            # mmcv.imwrite(expended_gt[j], save_path + type + '_' + str(num) + '.png')
            num += 1

# 生成切片数据并存储
# def createdData(X, label):
#     for c in range(OUTPUT_CLASSES):
#         PATCH, LABEL, TEST_PATCH, TRAIN_PATCH, TEST_LABEL, TRAIN_LABEL = [], [], [], [], [], []
#         for h in range(X.shape[1]-PATCH_SIZE+1):
#             print('step:', h)
#             for w in range(X.shape[2]-PATCH_SIZE+1):
#                 gt = label[h, w]
#                 if(gt == c+1):
#                     img = patch(X, PATCH_SIZE, h, w)
#                     PATCH.append(img)
#                     LABEL.append(gt-1)
#         # 打乱切片
#         shuffle(PATCH)
#         # 划分测试集与训练集
#         split_size = int(len(PATCH)*TEST_FRAC)
#         TEST_PATCH.extend(PATCH[:split_size])  # 0 ~ split_size
#         TRAIN_PATCH.extend(PATCH[split_size:])  # split_size ~ len(class)
#         TEST_LABEL.extend(LABEL[:split_size])
#         TRAIN_LABEL.extend(LABEL[split_size:])
#         # 写入文件夹
#         train_dict, test_dict = {}, {}
#         train_dict["train_patches"] = TRAIN_PATCH
#         train_dict["train_labels"] = TRAIN_LABEL
#         file_name = "Training_class(%d).mat" % c
#         scipy.io.savemat(os.path.join(NEW_DATA_PATH, file_name), train_dict)
#         test_dict["testing_patches"] = TEST_PATCH
#         test_dict["testing_labels"] = TEST_LABEL
#         file_name = "Testing_class(%d).mat" % c
#         scipy.io.savemat(os.path.join(NEW_DATA_PATH, file_name), test_dict)


# data, label = loadData("PaviaU", "PaviaU.mat", "PaviaU_gt.mat")
# data = standartize(data)
# data = pad(data, int((PATCH_SIZE-1)/2))
# createdData(data, label)


def main():
    img_size = 224
    new_data_path = os.path.join(ROOT_PATH, 'patch', DATA_TYPE + str(img_size))  # 存放数据路径 patch是文件夹名称
    print(new_data_path)
    img, gt = load_data(ROOT_PATH, DATA_TYPE)
    img = standartize(img)
    # img = pad(img, int((PATCH_SIZE - 1) / 2))
    # print(img.shape)
    # createdData(img, gt)
    crop_img(img, gt, DATA_TYPE, new_data_path, img_size=img_size)


if __name__ == '__main__':
    main()
