from torch.utils.data import Dataset
from pathlib import Path
from torchvision.io import read_image
from PIL import Image
import torch

class AutoDataset(Dataset):
    def __init__(self, folders, transforms=None):
        self.folders = folders
        self.transforms = transforms
        # create list of files
        self.f_list = []
        for i in self.folders:
            self.f_list += list(Path(i).rglob('*.jpg'))

    def __len__(self):
        return len(self.f_list)

    def __getitem__(self, idx):
        fname = self.f_list[idx]
        # image = read_image(fname)
        image = Image.open(fname)
        fname_split = str(fname).split("/")[-1].split("_")
        angle = int(fname_split[2].split(".jpg")[0])/1000.0
        velocity = int(fname_split[1])/1000.0
        if self.transforms is not None:
            image = self.transforms(image)
        return image.float() if isinstance(image, torch.tensor) else image, torch.tensor(angle, dtype=torch.float), torch.tensor(velocity, dtype=torch.float)