from torch.utils.data import Dataset
from pathlib import Path
from torchvision.io import read_image
from PIL import Image
from PIL import ImageOps
import torch
import random

class AutoDataset(Dataset):
    def __init__(self, folders, transforms=None, mirror_prob=0.5):
        self.folders = folders
        self.transforms = transforms
        self.mirror_prob = mirror_prob
        # create list of files
        self.f_list = []
        for i in self.folders:
            to_append = list(Path(i).rglob('*.jpg'))
            self.f_list += list(map(lambda e: e[1], sorted(zip(map(lambda fname: int(str(fname).split("/")[-1].split("_")[0]), to_append), to_append))))

        self.f_list =list(filter(lambda fname: int(str(fname).split("/")[-1].split("_")[1]) != 0, self.f_list))



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
