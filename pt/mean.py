from dataset import AutoDataset
import numpy as np
import torchvision as tv
d = AutoDataset([f"/share/user{i}" for i in range(5, 23)])

mu_images = np.zeros(3)
sigma_images = np.zeros(3)
counter = 0

for i in d:
    img, angle, speed = i
    transformer = tv.transforms.ToTensor()
    img_as_array = transformer(img)
    mean_of_image = np.mean(img_as_array, axis=(0, 1))
    np.add(mu_images, mean_of_image)
    counter = i

mu = mu_images / (counter + 1)
mu_squared = mu*mu

for i in d:
    img, angle, speed = i
    transformer = tv.transforms.ToTensor()
    img_as_array = transformer(img)
    img_as_array_squared = np.square(img_as_array)
    img_as_array_squared_minus = np.subtract(img_as_array_squared, mu_squared)
    sigma_of_image = np.mean(img_as_array_squared_minus, axis=(0, 1))
    np.add(sigma_images, sigma_of_image)


sigma = sigma_images / (counter + 1)

print(mu, sigma)
