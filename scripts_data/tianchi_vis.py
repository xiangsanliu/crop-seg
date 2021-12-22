from PIL import Image
import os


Image.MAX_IMAGE_PIXELS = 1000000000000

dir = "dataset/archive/jingwei_round2_train_20190726"


def tianchi_vis(image_name):
    image_path = os.path.join(dir, image_name)
    image = Image.open(image_path)
    width, height = image.width // 100, image.height // 100
    image.resize((width, height)).save(os.path.join(dir, "vis", image_name))


if __name__ == "__main__":
    for file_name in os.listdir(dir):
        if file_name.endswith(".png") and "label" not in file_name:
            print(f"{file_name} is processing...")
            tianchi_vis(file_name)
