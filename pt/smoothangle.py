import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# folders = [f"/home/tobi/fa/data/train{i}" for i in [4, 5, 6, 7]]
folders = [f"/home/tobi/fa/data/train{i}" for i in [7]]

folders = list(filter(lambda p: os.path.exists(p), folders))

f_list = []
for i in folders:
    f_list += list(Path(i).rglob('*.jpg'))

pic_nr = list(map(lambda fname: int(str(fname).split("/")[-1].split("_")[0].split(".")[0]), f_list))
angle = list(map(lambda fname: int(str(fname).split("/")[-1].split("_")[2].split(".")[0]), f_list))

fig, ax = plt.subplots(dpi=400)
ax.plot(pic_nr, angle, 'x')

ax.set(xlabel='driving time(pic_nr)', ylabel='angel')
ax.grid()

fig.savefig("angle.png")
plt.show()