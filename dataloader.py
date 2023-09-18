import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import os
from torchvision.io import read_image
from constants import *


def get_dataset(path):
    names = os.listdir(path)[:100]
    images = np.zeros((1, IMG_SIZE, IMG_SIZE))
    for name in names:
        images = np.concatenate((images, read_image(path + '/' + name)), axis=0)
    images = images[1:]
    images = images.reshape(100, -1)
    return images


hands = get_dataset('dataset/archive/Hand')
cxr = get_dataset('dataset/archive/CXR')
heads = get_dataset('dataset/archive/HeadCT')

dataset = np.concatenate((hands, cxr, heads), axis=0)
data_list = list([hands, cxr, heads])

