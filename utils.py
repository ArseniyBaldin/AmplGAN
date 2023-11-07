import torch
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


def load_weights(cls, epoch, batch_idx):
    with open('weights/model-{}-{}-{}.txt'.format(cls, epoch, batch_idx), 'r') as f:
        disc_w, gen_w = f.readlines()
        disc_w = torch.tensor(list(map(float, disc_w.split(' ')[:-1])), requires_grad=True)
        gen_w = torch.tensor(list(map(float, gen_w.split(' ')[:-1])), requires_grad=True)
    return disc_w, gen_w

def save_weights(disc, gen, cls, epoch, batch_idx):
    with open('weights/model-{}-{}-{}.txt'.format(cls, epoch, batch_idx), 'w') as f:
        for value in disc:
            f.write('{} '.format(value))
        f.write('\n')
        for value in gen:
            f.write('{} '.format(value))