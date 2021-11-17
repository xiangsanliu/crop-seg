import numpy as np
import torchvision.transforms.functional as TF


def to_tensor(sample):
    """Convert a sample to tensor"""
    image = sample["image"]
    sample["image"] = TF.to_tensor(image)

    if "mask" in sample:
        mask = sample["mask"]
        mask = TF.to_tensor(mask)
        sample["mask"] = mask
    return sample


def normalize(sample, mean, std, inplace=False):
    """Normalize a sample"""
    image = sample["image"]
    image = TF.normalize(image, mean, std, inplace=inplace)
    sample["image"] = image
    return sample


def hflip(sample):
    """Flip a sample horizontally"""
    image = sample["image"]
    sample["image"] = TF.hflip(image)
    if "mask" in sample:
        mask = sample["mask"]
        sample["mask"] = TF.hflip(mask)
    return sample


def vflip(sample):
    """Flip a sample vertically"""
    image = sample["image"]
    sample["image"] = TF.vflip(image)
    if "mask" in sample:
        mask = sample["mask"]
        sample["mask"] = TF.vflip(mask)
    return sample

def random_rot(sample):
    """Random rotate a sample"""
    angles = [0, 90, 180, 270]
    angle = np.random.choice(angles)
    image = sample["image"]
    sample["image"] = TF.rotate(image, angle)
    if "mask" in sample:
        mask = sample["mask"]
        sample["mask"] = TF.rotate(mask, angle)
    return sample

def random_crop(sample, size):
    """Random crop a sample"""

    image = sample["image"]
    image_h, image_w = image.shape[:2]
    crop_h, crop_w = size
    topleft_h = np.random.randint(0, image_h - crop_h)
    topleft_w = np.random.randint(0, image_w - crop_w)

    sample["image"] = TF.crop(image, topleft_h, topleft_w, crop_h, crop_w)
    if "mask" in sample:
        mask = sample["mask"]
        sample["mask"] = TF.crop(mask, topleft_h, topleft_w, crop_h, crop_w)
    return sample

def center_crop(sample, size):
    """Center crop a sample"""

    image = sample["image"]

    sample["image"] = TF.center_crop(image, size)
    if "mask" in sample:
        mask = sample["mask"]
        sample["mask"] = TF.center_crop(mask, size)
    return sample

def adjust_brightness(img, factor):
    """Adjust brightness of an image"""
    return TF.adjust_brightness(img, factor)

def adjust_contrast(img, factor):
    """Adjust contrast of an image"""
    return TF.adjust_contrast(img, factor)

def adjust_saturation(img, factor):
    """Adjust saturation of an image"""
    return TF.adjust_saturation(img, factor)

def adjust_hue(img, factor):
    """Adjust hue of an image"""
    return TF.adjust_hue(img, factor)
    