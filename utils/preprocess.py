"""
This script aims to build the kerneled labels for Corneal Nerve training.
    The original labels only contains a centerline of the the tree-like structures.
    In order to handle the imbalance samples, we expand the centerline with
    a 2x2 kernel.
"""

import os
import glob
from PIL import Image
import cv2
import numpy as np


def convert_img(image):
    image = [255 if image == True else 0]
    return image


def expand_label(path, save_path):
    """
    expand the centerline with a 2x2 kernel
    :param path: source label path
    :param save_path: target label path
    :return: None
    """
    for file in glob.glob(os.path.join(path, '*skelton.png')):
        print(file)
        base_name = os.path.basename(file)
        image = cv2.imread(file)
        kernel = np.ones((3, 3), np.uint8)
        img = cv2.dilate(image, kernel, iterations=1)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
        img = Image.fromarray(img)
        img.save(save_path + base_name)

