from PIL import Image
import os
import random
import shutil
import numpy as np


Image.MAX_IMAGE_PIXELS = 1000000000000


def tianchi_vis(dir, image_name):
    image_path = os.path.join(dir, image_name)
    image = Image.open(image_path)
    width, height = image.width // 100, image.height // 100
    image.resize((width, height)).save(os.path.join(dir, "vis", image_name))


def vis_all(dir):
    for file_name in os.listdir(dir):
        if file_name.endswith(".png") and "label" not in file_name:
            print(f"{file_name} is processing...")
            tianchi_vis(dir, file_name)


def random_copy(image_dir, mask_dir):
    selcect_list = []
    file_names = os.listdir(image_dir)
    while len(selcect_list) < 6:
        file_name = file_names[int(len(file_names) * random.random())]
        selcect_list.append(file_name)
        # copy to vis
        shutil.copy(
            os.path.join(image_dir, file_name),
            os.path.join(image_dir, "vis", file_name),
        )
        mask_vis = vis_mask(os.path.join(mask_dir, file_name))
        print(mask_vis.shape)
        mask = Image.fromarray(np.uint8(mask_vis))
        mask.save(os.path.join(mask_dir, "vis", file_name))


def vis_mask(mask_path):
    mask = Image.open(mask_path)
    mask = np.array(mask)
    mask_vis = np.zeros((mask.shape[0], mask.shape[1], 3)) + 255
    background = mask == 0
    tobacco = mask == 1
    corn = mask == 2
    barley = mask == 3
    building = mask == 4
    mask_vis[:, :, 0][background] = 0
    mask_vis[:, :, 1][background] = 0
    mask_vis[:, :, 2][background] = 0

    # red
    mask_vis[:, :, 0][tobacco] = 255
    mask_vis[:, :, 1][tobacco] = 0
    mask_vis[:, :, 2][tobacco] = 0

    # gleen
    mask_vis[:, :, 0][corn] = 0
    mask_vis[:, :, 1][corn] = 255
    mask_vis[:, :, 2][corn] = 0

    # blue
    mask_vis[:, :, 0][barley] = 0
    mask_vis[:, :, 1][barley] = 0
    mask_vis[:, :, 2][barley] = 255

    # gray
    mask_vis[:, :, 0][building] = 128
    mask_vis[:, :, 1][building] = 128
    mask_vis[:, :, 2][building] = 128

    return mask_vis


if __name__ == "__main__":
    # random_copy(image_dir="datasets/tianchi/round2_no_overlap/image", mask_dir="datasets/tianchi/round2_no_overlap/label")
    # vis_all("datasets/archive/jingwei_round1_train_20190619")
    mask_vis = vis_mask("datasets/tianchi/round2_no_overlap/label/image_21_2218.png")
    mask = Image.fromarray(np.uint8(mask_vis))
    mask.save(os.path.join("scripts_data", "a.png"))
