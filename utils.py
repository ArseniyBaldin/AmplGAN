from pennylane import numpy as np
import matplotlib.pyplot as plt
from dataloader import *
from constants import *


def make_noise(scale=NOISE_SCALE, num=NOISE_SIZE):
    noise = 1 / (2 * np.sqrt(PCA_DIM)) + np.random.normal(0, scale / (2 * np.sqrt(PCA_DIM)), size=(num, PCA_DIM))
    noise = project2sphere(stretch(noise))
    return torch.tensor(noise)


def init_random_variables(length, val, grad=True):
    par = torch.rand(length) * val
    par.requires_grad_(grad)
    return par
