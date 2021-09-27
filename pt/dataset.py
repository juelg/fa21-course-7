from torch.utils.data import Dataset
from pathlib import Path
from torchvision.io import read_image


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

    def __iter__(self):
        for i in self.f_list:
            image = read_image(i)
            label = i.split("_")[2]
            if self.transforms is not None:
                image = self.transforms(image)
            yield image, label