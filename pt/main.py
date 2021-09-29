from dataset import AutoDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pl_module import AutoModule
import os
from torchvision.transforms import transforms
import numpy as np
from torchvision.transforms.functional import crop
import torch

hparams = {"learning_rate": 1e-3, "batch_size": 64, "weight_decay": 1e-6, "img_size": 256, "workers": 6,
    "noise": None, "zero_angle_dropout": False, "large_angle_dropout": False, "zero_angle_dropout2": False}
gpu = 1
epochs = 10

std = np.array([50.2475, 47.8637, 56.0220])/255
mean = np.array([136.2113, 141.1734, 130.5111])/255

class LambdaTrans():
    def __init__(self, f):
        self.f = f

    def __call__(self, x):
        return self.f(x)


transforms_compose = transforms.Compose([
        # transforms.Resize((hparams["img_size"], hparams["img_size"])), # todo: scaling!
        LambdaTrans(lambda x: crop(x, 160-100, 0, 100, 320)), # crop out the upper part of the image -> 72 x 320
        # transforms.ColorJitter(brightness=np.random.uniform(0.4, 0.6), contrast=np.random.uniform(0.7, 0.9),
        #                    saturation=np.random.uniform(0.7, 0.9)),
        transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur(9, sigma=(0.1, 2))]), p=0.8),
        
        transforms.ToTensor(),
        # transforms.Normalize(mean, std),

    # LambdaTrans(lambda x: x/255)
    ])

folders = [f"/share/user{i}" for i in set(range(6, 23))-set([8])] + [f"/share2/user{i}" for i in set(range(6, 23))-set([8])]
# folders = ["/home/tobi/fa/imgs"] #+ 
# folders = [f"/home/tobi/fa/data/train{i}" for i in [4, 5]]
# folders += [f"/home/tobi/fa/data/center/train{i}" for i in [6, 7]]

folders = list(filter(lambda p: os.path.exists(p), folders))

if __name__ == "__main__":


    data = AutoDataset(folders, transforms_compose, zero_angle_dropout=hparams["zero_angle_dropout"], large_angle_dropout=hparams["large_angle_dropout"], zero_angle_dropout2=hparams["zero_angle_dropout2"])
    #for i in data:
    #    print(i[1])
    #    i[0].save("asdf43.png")
    #    exit()
        
    pl_module = AutoModule(hparams, data)


    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        strict=True,
        verbose=True,
        mode='min'
    )
    model_checkpoint = ModelCheckpoint(
                monitor='val_loss',
                save_last=True,
                save_top_k=1,
    )
    # callbacks = [early_stop_callback, model_checkpoint]
    callbacks = [model_checkpoint]

    trainer = pl.Trainer(
        #row_log_interval=1,
        #track_grad_norm=2,
        # weights_summary=None,
        #distributed_backend='dp',
        callbacks=callbacks,
        max_epochs=epochs,
        deterministic=True,
        #profiler=True,
        #fast_dev_run=True,
        gpus=[gpu], #[0, 1],
        default_root_dir="lightning_logs2" #os.path.join(results_path, "supervised", "loss_dist_sphere_fix_radius", "asdf"),
        #auto_select_gpus=True,
        #enable_pl_optimizer=True,
    )
    trainer.fit(pl_module)
