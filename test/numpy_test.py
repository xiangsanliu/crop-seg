from PIL import Image
import numpy as np

def cal_vdvi(image):
    pass



def cal(image_path):
    image = Image.open(image_path).convert("RGB")
    image = np.asarray(image)


def calc_indices(image_path):
    image = Image.open(image_path).convert("RGB")
    image = np.asarray(image)
    r, g, b = image.split()
    r = np.asarray(r)
    g = np.asarray(g)
    b = np.asarray(b)
    R = r / (r + g + b)
    G = g / (r + g + b)
    B = b / (r + g + b)
    exg = 2 * G - R - B    
    denominator = 2 * G + R + B
    vdvi = exg / denominator
    
    
    