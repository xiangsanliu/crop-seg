import numpy as np
import pandas as pd
import cv2 as cv
from PIL import Image
from tqdm import tqdm
import os
from argparse import ArgumentParser
from multiprocessing import Pool


if __name__ == "__main__":

    parser = ArgumentParser(description="")
    parser.add_argument("-image_path", type=str)
    parser.add_argument("-save_dir", type=str, default=r"./data/")
    parser.add_argument("-type", type=str, default='image')
    parser.add_argument("-suffix", type=str, default="1")
    arg = parser.parse_args()
    image_path = arg.image_path
    save_image_dir = os.path.join(arg.save_dir, arg.type)
    stride = 512
    target_size = (512, 512)

    if not os.path.isdir(save_image_dir):
        os.makedirs(save_image_dir)
    root_dir, filename = os.path.split(image_path)
    basename, filetype = os.path.splitext(filename)
    print(basename)

    image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
    image = np.asarray(image)
    print("Image loaded!")
    print("Image size:", image.shape)
    cnt = 0
    csv_pos_list = []

    # 裁剪边界至步长整数倍,方便整除
    target_w, target_h = target_size
    h, w = image.shape[0], image.shape[1]

    new_w = w - w % target_w
    new_h = h - h % target_h
    image = image[0:new_h, 0:new_w]
    print("New Image Size: ", image.shape)

    def crop(cnt, crop_image):
        image_name = os.path.join(save_image_dir, f"{cnt}_{arg.suffix}.png")
        cv.imwrite(image_name, crop_image)

    h, w = image.shape[0], image.shape[1]
    for i in tqdm(range(w // stride - 1)):
        for j in range(h // stride - 1):
            topleft_x = i * stride
            topleft_y = j * stride
            crop_image = image[
                topleft_y : topleft_y + target_h, topleft_x : topleft_x + target_w
            ]

            if crop_image.shape[:2] != (target_h, target_h):
                print(topleft_x, topleft_y, crop_image.shape)

            else:
                crop(cnt, crop_image)
                csv_pos_list.append(
                    [
                        basename + "_" + str(cnt) + ".png",
                        topleft_x,
                        topleft_y,
                        topleft_x + target_w,
                        topleft_y + target_h,
                    ]
                )
                cnt += 1
    csv_pos_list = pd.DataFrame(csv_pos_list)
    csv_pos_list.to_csv(
        os.path.join(arg.save_dir, basename + ".csv"), header=None, index=None
    )
