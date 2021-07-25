import os
from tools.data import load_data, random_patch, random_crop


# 定义全局变量
PATCH_SIZE = 17  # 切片尺寸
OUTPUT_CLASSES = 16  # 输出9类地物
TEST_FRAC = 0.50  # 用来测试数据的百分比
ROOT_PATH = '/home/xiangjianjian/dataset/WHU-Hi/'
DATA_TYPE = 'WHU_Hi_HanChuan'

def main():
    img_size = 224
    new_data_path = os.path.join(ROOT_PATH, 'patch', DATA_TYPE + str(img_size))  # 存放数据路径 patch是文件夹名称
    print(new_data_path)
    img, gt = load_data(ROOT_PATH, DATA_TYPE)
    
    # img = pad(img, int((PATCH_SIZE - 1) / 2))
    # print(img.shape)
    # createdData(img, gt)
    # crop_img(img, gt, DATA_TYPE, new_data_path, img_size=img_size)
    print(new_data_path)
    random_crop(img, gt, DATA_TYPE, new_data_path + 'random',img_size=img_size)


if __name__ == '__main__':
    main()
