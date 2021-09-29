from torch.utils.data import Dataset
from pathlib import Path
from torchvision.io import read_image
from PIL import Image
from PIL import ImageOps
import torch
import random

class AutoDataset(Dataset):
    def __init__(self, folders, transforms=None, mirror_prob=0.5, zero_angle_dropout=False, zero_angle_dropout2=False, large_angle_dropout=False):
        self.folders = folders
        self.transforms = transforms
        self.mirror_prob = mirror_prob
        self.zero_angle_dropout = zero_angle_dropout
        self.zero_angle_dropout2 =zero_angle_dropout2
        self.large_angle_dropout = large_angle_dropout
        self.file_ending = ".jpg"
        # create list of files
        self.f_list = []
        for i in self.folders:
            # print(i)
            to_append = list(Path(i).rglob(f'*{self.file_ending}'))
            if len(to_append) == 0:
                continue
            to_append = sorted(list(zip(map(lambda fname: list(map(int, str(fname).replace(self.file_ending, '').split("/")[-1].split("_"))), to_append), to_append)))

            for i in range(len(to_append)):
                to_append[i][0][0] = i

            minimum_idx = min(map(lambda entry: entry[0][0], to_append[:]))
            maximum_idx = max(map(lambda entry: entry[0][0], to_append[:]))



            if self.zero_angle_dropout:
                threshold_left = 5
                threshold_right = 5

                filter_func_zero = lambda entry: \
                    entry[0][2] != 0 or \
                    len(list(filter(lambda e: e == 0, [to_append[max(entry[0][0] - i, minimum_idx)] for i in range(1, threshold_left + 1)]))) < threshold_left or \
                    len(list(filter(lambda e: e == 0, [to_append[min(entry[0][0] + i, maximum_idx)] for i in range(1, threshold_right+1)]))) < threshold_right
 
                to_append = list(filter(filter_func_zero, to_append))

            if self.zero_angle_dropout2:
                to_append = list(filter(lambda entry: entry[0][2] != 0 and random.random() < 0.8, to_append))

            if self.large_angle_dropout:
                filter_func_large = lambda entry: abs(entry[0][2]) <= 900
                to_append = list(filter(filter_func_large, to_append))

            self.f_list += list(map(lambda e: e[1], sorted(to_append)))



    def __len__(self):
        #print(len(self.f_list))
        return len(self.f_list)

    def __getitem__(self, idx):
        mirror = random.random() < self.mirror_prob

        fname = self.f_list[idx]
        #print("Filename:", fname)
        image = Image.open(fname)
        fname_split = str(fname).split("/")[-1].split("_")
        angle = int(fname_split[2].split(self.file_ending)[0])/1000.0
        #angle=1.0
        #print("Angle:", angle)
        velocity = int(fname_split[1])/1000.0

        if mirror:
            image = ImageOps.mirror(image)
            angle *= -1

        if self.transforms is not None:
            image = self.transforms(image)
        return image.float() if isinstance(image, torch.Tensor) else image, torch.tensor(angle, dtype=torch.float), torch.tensor(velocity, dtype=torch.float)
