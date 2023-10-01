# from dataloader import *
import numpy as np
import matplotlib.pyplot as plt

ax = plt.figure().add_subplot(projection='3d')
ax.set_zlim(0, 1)


def rotation_animation(data, num):
    for theta in np.linspace(0, 360, num):
        ax.view_init(theta / 3, theta)
        ax.scatter(data[:, 0], data[:, 1], data[:, 2])
        plt.savefig("rotation_imgs/{}.png".format(int(theta)))
        ax.cla()


def animation(data, func, num_frames, order, thetas):
    for num, t in enumerate(np.linspace(0, 1, num_frames)):
        new_data = func(data, t)
        # print(new_data.shape)
        if new_data.shape[1] == 2:
            ax.scatter(new_data[:, 0], new_data[:, 1], zs=0, alpha=1, s=2)
        else:
            ax.scatter(new_data[:, 0], new_data[:, 1], new_data[:, 2], alpha=1, s=2)
        ax.view_init(30, thetas[0] + (thetas[1] - thetas[0]) * t)
        ax.set_zlim(0, 1)
        plt.savefig("transform_animation/{}_{}.png".format(order, num))
        ax.cla()


def full_animation(data):
    animation(data, translation, 100, 0, [0, 0])
    animation(translation(data), stretch, 100, 1, [0, 0])
    animation(stretch(translation(data)), project2sphere, 100, 2, [0, 360])
    animation(project2sphere(stretch(translation(data))), inverse_project2sphere, 100, 3, [0, 360])
    animation(inverse_project2sphere(project2sphere(stretch(translation(data)))), inverse_stretch, 100, 4, [0, 0])
    animation(inverse_stretch(inverse_project2sphere(project2sphere(stretch(translation(data))))), inverse_translation,
              100, 5, [0, 0])
