from pennylane import numpy as np

IMG_SIZE = 64
PCA_DIM = 2
N_dim = PCA_DIM
N_a = 2
DEPTH = 2
NUM_LAYER_WEIGHTS = 3 * (N_a + N_dim) - 1
NUM_DEEP_LAYER_WEIGHTS = NUM_LAYER_WEIGHTS * DEPTH
NOISE_SCALE = 0.05
NOISE_SIZE = 100