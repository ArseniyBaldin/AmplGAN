from pennylane import numpy as np
import matplotlib.pyplot as plt
from dataloader import *
from constants import *


def make_noise(size=NOISE_SCALE, num=NOISE_SIZE):
    noise = np.random.normal(1 / (2 * np.sqrt(PCA_DIM)), size / (2 * np.sqrt(PCA_DIM)), size=(num, PCA_DIM))
    noise = project2sphere(stretch(noise))
    return noise

