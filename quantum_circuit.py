import pennylane as qml
from pennylane import numpy as np
from constants import *
import torch

dev = qml.device('lightning.qubit', wires=2 * (N_a + N_dim) + 1)


def single(start, end, w):
    # print('single: ', w)
    for weight_index, wire in enumerate(range(start, end + 1)):
        qml.RY(w[weight_index], wires=wire)


def dual(start, end, w):
    # print('dual: ', w)
    for weight_index, wire in enumerate(range(start, end)):
        qml.IsingXY(w[weight_index], wires=[wire, wire + 1])


def entangle(start, end, w):
    # print('entangle: ', w)
    for weight_index, wire in enumerate(range(start, end)):
        qml.CRY(w[weight_index], wires=[wire, wire + 1])
    qml.CRY(w[-1], wires=[end, start])


def hadamard_column(start, end):
    for wire in range(start, end + 1):
        qml.Hadamard(wires=wire)


def input_data(data):
    single(2 * N_a + N_dim + 1, 2 * (N_a + N_dim), data)


def embed(data):
    qml.AmplitudeEmbedding(features=data, wires=range(2 * N_a + N_dim + 1, 2 * (N_a + N_dim) + 1), normalize=True)


def swap_test():
    qml.Hadamard(wires=0)
    for i in range(N_dim):
        qml.CSWAP(wires=[0, N_a + 1 + i, 2 * N_a + N_dim + 1 + i])
    qml.Hadamard(wires=0)


def layer(start, end, w):
    # print('layer: ', w)
    length = end - start + 1
    single(start, end, w[:length])
    dual(start, end, w[length:2 * length - 1])
    entangle(start, end, w[2 * length - 1:3 * length - 1])


def deep_layer(start, end, repeat, w):
    # print('deep_layer: ', w)
    hadamard_column(start, end)
    for i in range(repeat):
        layer(start, end, w[NUM_LAYER_WEIGHTS * i:NUM_LAYER_WEIGHTS * (i + 1)])


@qml.qnode(dev)
def gen_sample(disc_weights, gen_weights, noise):
    embed(noise)
    deep_layer(1, N_a + N_dim, DEPTH, disc_weights)
    deep_layer(N_a + N_dim + 1, 2 * (N_a + N_dim), DEPTH, gen_weights)
    return qml.probs(wires=range(2 * N_a + N_dim + 1, 2 * (N_a + N_dim) + 1))


@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def train_on_fake(disc_weights, gen_weights, noise):
    embed(noise)
    deep_layer(1, N_a + N_dim, DEPTH, disc_weights)
    deep_layer(N_a + N_dim + 1, 2 * (N_a + N_dim), DEPTH, gen_weights)
    swap_test()
    return qml.expval(qml.PauliZ(0))


@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def train_on_real(disc_weights, data):
    embed(data)
    deep_layer(1, N_a + N_dim, DEPTH, disc_weights)
    swap_test()
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def test(data):
    embed(data)
    return qml.probs(wires=range(2 * N_a + N_dim + 1, 2 * (N_a + N_dim) + 1))

