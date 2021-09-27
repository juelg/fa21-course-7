from pathlib import Path
import PIL.Image


def load_image(path):
    filename = Path(path).stem.split('_')
    return PIL.Image.open(path), (filename[1], filename[2])
