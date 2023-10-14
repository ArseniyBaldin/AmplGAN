import statistics

import matplotlib.pyplot as plt
import torch

from dataloader import *
from loss_utils import *
from utils import *
from constants import *
import torch.nn as nn
import torch.nn.utils as torch_utils

noise = make_noise()

disc_weights = init_random_variables(NUM_DEEP_LAYER_WEIGHTS, np.pi/2)
gen_weights = init_random_variables(NUM_DEEP_LAYER_WEIGHTS, np.pi/2)

criterion = nn.BCELoss()
disc_optimizer = torch.optim.AdamW([disc_weights], lr=LEARNING_RATE, betas=(0.5, 0.999))
gen_optimizer = torch.optim.AdamW([gen_weights], lr=LEARNING_RATE, betas=(0.5, 0.999))



def train_amplitude():
    for num_cls, cls in enumerate(ampl_list):
        train_data = torch.utils.data.DataLoader(cls, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
        for epoch in range(NUM_EPOCHS):

            for batch_idx, batch in enumerate(train_data):

                # print(batch)
                # sample = torch.sqrt(gen_sample(disc_weights, gen_weights,
                #                     torch.from_numpy(np.random.uniform(0, NOISE_SCALE, [1, LATENT_DIM + 1]))).reshape(1, -1)).detach().numpy()
                # sample = hat2latent(sample)
                # show_img(epoch, batch_idx, vec=pca.inverse_transform(scale.inverse_transform(sample)))
                # fig, ax = qml.draw_mpl(Gen_sample)(disc_weights, gen_weights, hat_noise)
                # fig.show()
                sample = torch.sqrt(gen_sample(disc_weights, gen_weights, make_noise()))
                sample = inverse_translation(inverse_stretch(inverse_project2sphere(sample)))
                sample = pca.inverse_transform(sample.reshape(1, -1)).reshape(64, 64)
                plt.imshow(sample, origin='lower', cmap='gray')
                plt.savefig("gen_med/image-{}-{}".format(epoch, batch_idx))
                plt.clf()

                for iter in range(A):
                    disc_optimizer.zero_grad()
                    loss = Batch_real_loss(disc_weights, batch)
                    loss.backward()
                    # torch_utils.clip_grad_norm_(disc_weights, max_norm=max_norm)
                    disc_optimizer.step()

                for iter in range(B):
                    disc_optimizer.zero_grad()
                    loss = Batch_fake_loss(disc_weights, gen_weights, make_noise())
                    loss.backward()
                    # torch_utils.clip_grad_norm_(disc_weights, max_norm=max_norm)
                    disc_optimizer.step()

                for iter in range(C):
                    gen_optimizer.zero_grad()
                    loss = Batch_gen_loss(disc_weights, gen_weights, make_noise())
                    loss.backward()
                    # print(torch.gradient(gen_weights))
                    grad_norm = (torch.sqrt(torch.sum(torch.pow(gen_weights.grad, 2)))).item()
                    # torch_utils.clip_grad_norm_(disc_weights, max_norm=max_norm)
                    gen_optimizer.step()


train_amplitude()
# aboba = pca.inverse_transform(inverse_translation(inverse_stretch(inverse_project2sphere(torch.sqrt(test(ampl_list[0][0])))))).reshape(64,64)
# plt.imshow(aboba)
# plt.show()