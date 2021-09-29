from torch.utils.data import Dataset
from pathlib import Path
from torchvision.io import read_image
from PIL import Image
from PIL import ImageOps
import torch
import random

class AutoDataset(Dataset):
    def __init__(self, folders, transforms=None, mirror_prob=0.5, zero_angle_dropout=False, large_angle_dropout=False):
        self.folders = folders
        self.transforms = transforms
        self.mirror_prob = mirror_prob
        self.zero_angle_dropout = zero_angle_dropout
        self.large_angle_dropout = large_angle_dropout
        # create list of files
        self.f_list = []
        for i in self.folders:
            to_append = list(Path(i).rglob('*.jpg'))
            to_append = zip(map(lambda fname: list(map(int, str(fname).replace('.jpg', '').split("/")[-1].split("_")), to_append)), to_append)

            minimum_idx = min(map(lambda entry: entry[0][0], to_append))
            maximum_idx = max(map(lambda entry: entry[0][0], to_append))

            if self.zero_angle_dropout:
                threshold_left = 2
                threshold_right = 2

                filter_func_zero = lambda entry: \
                    entry[0][2] != 0 or \
                    len(list(filter(lambda e: e == 0, [to_append[max(entry[0][0] - i, minimum_idx)] for i in range(1, threshold_left + 1)]))) < threshold_left or \
                    len(list(filter(lambda e: e == 0, [to_append[min(entry[0][0] + i, maximum_idx)] for i in range(1, threshold_right+1)]))) < threshold_right

                to_append = list(filter(filter_func_zero, to_append))

            if self.large_angle_dropout:
                filter_func_large = lambda entry: abs(entry[0][2]) <= 900
                to_append = list(filter(filter_func_large, to_append))

            self.f_list += list(map(lambda e: e[1], sorted(to_append)))



    def __len__(self):
        return len(self.f_list)

    def __getitem__(self, idx):
        mirror = random.random() < self.mirror_prob

        fname = self.f_list[idx]
        # image = read_image(fname)
        image = Image.open(fname)
        fname_split = str(fname).split("/")[-1].split("_")
        angle = int(fname_split[2].split(".jpg")[0])/1000.0
        velocity = int(fname_split[1])/1000.0

        if mirror:
            image = ImageOps.mirror(image)
            angle *= -1

        if self.transforms is not None:
            image = self.transforms(image)
        return image.float() if isinstance(image, torch.Tensor) else image, torch.tensor(angle, dtype=torch.float), torch.tensor(velocity, dtype=torch.float)
