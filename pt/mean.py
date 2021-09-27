from dataset import AutoDataset
import numpy as np
import torchvision as tv
import os


def get_sigma_mu():
    folders = list(filter(lambda p: os.path.exists(p), [f"/share/user{i}" for i in range(6, 23)]))

    dataset = AutoDataset([f"/share/user{i}" for i in range(6, 23)])

    mu_images = np.zeros(3)
    sigma_images = np.zeros(3)
    counter = 0

    for entry in dataset:
        img, angle, speed = entry
        img_as_array = tv.transforms.ToTensor(img)
        mean_of_image = np.mean(img_as_array, axis=(0, 1))
        mu_images += mean_of_image
        counter += 1

    mu = mu_images / counter
    mu_squared = np.square(mu)

    for entry in dataset:
        img, angle, speed = entry
        img_as_array = tv.transforms.ToTensor(img)
        img_as_array_squared = np.square(img_as_array)
        img_as_array_squared_minus = np.subtract(img_as_array_squared, mu_squared)

        sigma_of_image = np.mean(img_as_array_squared_minus, axis=(0, 1))
        sigma_images += sigma_of_image

    sigma = sigma_images / counter

    return mu, sigma
