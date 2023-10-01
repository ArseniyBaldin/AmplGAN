import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import os
from torchvision.io import read_image
from constants import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from animation_utils import *


def get_dataset(path):
    names = os.listdir(path)[:1000]
    images = np.zeros((1, IMG_SIZE, IMG_SIZE))
    for name in names:
        images = np.concatenate((images, read_image(path + '/' + name)), axis=0)
    # images = images[1:]
    images = images.reshape(1001, -1)
    return images


hands = get_dataset('dataset/archive/Hand')
cxr = get_dataset('dataset/archive/CXR')
heads = get_dataset('dataset/archive/HeadCT')
dataset = np.concatenate((hands, cxr, heads), axis=0)
data_list = list([hands, cxr, heads])

pca = PCA(n_components=PCA_DIM)
pca.fit(dataset)

hands_pca = pca.transform(hands)
cxr_pca = pca.transform(cxr)
heads_pca = pca.transform(heads)
dataset_pca = np.concatenate((hands_pca, cxr_pca, heads_pca), axis=0)
data_pca_list = list([hands_pca, cxr_pca, heads_pca])

scaler = MinMaxScaler(feature_range=(-1 / np.sqrt(PCA_DIM), 1 / np.sqrt(PCA_DIM)))
scaler.fit(dataset_pca)

hands_scaled = scaler.transform(hands_pca)
cxr_scaled = scaler.transform(cxr_pca)
heads_scaled = scaler.transform(heads_pca)
dataset_scaled = np.concatenate((hands_scaled, cxr_scaled, heads_scaled), axis=0)
data_scaled_list = list([hands_scaled, cxr_scaled, heads_scaled])


def translation(data, t=1):
    transformed_data = scaler.transform(data)
    # print("translation: ", data.shape)
    return (1 - t) * data + t * transformed_data


def inverse_translation(data, t=1):
    transformed_data = scaler.inverse_transform(data)
#     print("inverse_translation: ", data.shape)
    return (1 - t) * data + t * transformed_data


def stretch(data, t=1):
    def transform(sample):
        trans_data = np.sqrt(PCA_DIM / np.sum(np.power(sample, 2))) * np.absolute(np.max(np.absolute(sample))) * sample
        vec_norm = np.sqrt(np.sum(np.power(trans_data, 2)))
        trans_data *= np.sin(vec_norm * np.arcsin(1)) / vec_norm
        return trans_data

    transformed_data = np.array([transform(sample) for sample in data])
#     print("stretch: ", data.shape)
    return (1 - t) * data + t * transformed_data


def inverse_stretch(data, t=1):
    def transform(sample):
        vec_norm = np.sqrt(np.sum(np.power(sample, 2)))
        if vec_norm > 1:
            vec_norm = 1
        trans_data = np.arcsin(vec_norm) / (np.arcsin(1) * vec_norm) * sample[:-1]
        trans_data = (np.sqrt(PCA_DIM / np.sum(np.power(trans_data, 2))) * np.absolute(
            np.max(np.absolute(trans_data)))) ** (-1) * trans_data
        return trans_data

    transformed_data = np.array([transform(sample) for sample in data])
#     print("inverse_stretch: ", transformed_data.shape)
    return (1 - t) * data[:, :-1] + t * transformed_data


def project2sphere(data, t=1):
    z_dim = np.sqrt(np.abs(1 - np.sum(np.power(data, 2), axis=1)).reshape(data.shape[0], 1))
    new_data = np.concatenate((data, z_dim), axis=1)
#     print("project2sphere: ", new_data.shape)

    return (1 - t) * np.concatenate((data, np.zeros(data.shape[0]).reshape(data.shape[0], 1)), axis=1) + t * new_data


def inverse_project2sphere(data, t=1):
    new_data = data.copy()
    new_data[:, PCA_DIM] = 0
#     print("inverse_project2sphere: ", new_data.shape)

    return (1 - t) * data + t * new_data


aboba = inverse_translation(inverse_stretch(inverse_project2sphere(project2sphere(stretch(translation(dataset_pca))))))
print(np.sum(np.power(aboba - dataset_pca, 2)))
