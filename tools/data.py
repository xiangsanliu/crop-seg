# -*- coding: utf-8 -*-
"""
Created on Wed May 15 17:17:30 2019

@author: viryl
"""
import os
import math
import random
import scipy.io
import scipy.ndimage
import torch
import numpy as np
from tqdm import tqdm
from tools.utils import pca, pad, standartize, patch
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
    img = standartize(img)
    gt = np.array(gt[data_type + '_gt'])
    return img, gt


def patch(img, gt, img_size, a, b):
    h, w = gt.shape
    b1 = b+img_size
    a1 = a+img_size
    patch_img = img[:, b:b1, a:a1]
    patch_gt = gt[b:b1, a:a1]
    c, h, w = patch_img.shape
    if h != img_size or w != img_size:
        # 补零
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


def sorted_patch(img, gt, img_size):
    h, w = gt.shape
    num_x = math.ceil(w / img_size)
    num_y = math.ceil(h / img_size)
    for j in range(num_y):
        for i in range(num_x):
            a = i * img_size
            b = j * img_size
            patched_img, patched_gt = patch(img, gt, img_size, a, b)


def random_patch(img, gt, img_size):
    h, w = gt.shape
    a = random.randint(0, w - img_size)
    b = random.randint(0, h - img_size)
    patch_img, patch_gt = patch(img, gt, img_size, a, b)
    patch_img = np.array(patch_img, dtype=np.float32)
    patch_gt = np.array(patch_gt, dtype=np.int64)
    return patch_img, patch_gt


def random_patch_torch(img, gt, img_size):
    patch_img, patch_gt = random_patch(img, gt, img_size)
    return torch.from_numpy(patch_img), torch.from_numpy(patch_gt)


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
    return [img]


def random_crop(img, gt, type, save_path, img_size=224, ratio=0.9):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    mats = []
    print('Cropping Image...')
    for i in tqdm(range(500)):
        patched_img, patched_gt = random_patch(img, gt, img_size)
        expanded_img = expand_data(patched_img)
        expanded_gt = expand_data(patched_gt)
        for j in range(len(expanded_img)):
            mat = {
                'img': np.transpose(expanded_img[j], (2, 0, 1)),
                'gt': expanded_gt[j]
            }
            mats.append(mat)
            # file_path = os.path.join(save_path, '%s_%03d.mat' % (type, num))
            # savemat(file_path, mat)
    split_data(mats, save_path, ratio)


def split_data(mats, save_path, ratio):
    random.shuffle(mats)
    total = len(mats)
    offset = int(total * ratio)
    train_mats = mats[:offset]
    val_mats = mats[offset:]
    print('Saving Train Images...')
    for i in tqdm(range(len(train_mats))):
        file_path = os.path.join(save_path, '%s_%03d.mat' % ('train', i))
        savemat(file_path, train_mats[i])
    print('Saving Valid Images...')
    for i in tqdm(range(len(val_mats))):
        file_path = os.path.join(save_path, '%s_%03d.mat' % ('val', i))
        savemat(file_path, val_mats[i])


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
