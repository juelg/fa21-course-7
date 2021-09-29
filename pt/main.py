from dataset import AutoDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pl_module import AutoModule
import os
from torchvision.transforms import transforms
import numpy as np
from torchvision.transforms.functional import crop

hparams = {"learning_rate": 1e-4, "batch_size": 32, "weight_decay": 1e-5, "img_size": 256, "workers": 4,
    "noise": None, "zero_angle_dropout": True, "large_angle_dropout": True}
gpu = 0
epochs = 50

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
        # transforms.ColorJitter(brightness=np.random.uniform(0.5), contrast=np.random.uniform(0.5),
        #                    saturation=np.random.uniform(0.5)),
        transforms.GaussianBlur(9, sigma=1),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std),

    # LambdaTrans(lambda x: x/255)
    ])

# folders = [f"/share/user{i}" for i in range(6, 23)]
# folders = ["/home/tobi/fa/imgs"] #+ 
folders = [f"/home/tobi/fa/data/train{i}" for i in [4, 5]]
folders += [f"/home/tobi/fa/data/center/train{i}" for i in [6, 7]]

folders = list(filter(lambda p: os.path.exists(p), folders))

if __name__ == "__main__":


    data = AutoDataset(folders, transforms_compose, zero_angle_dropout=hparams["zero_angle_dropout"], large_angle_dropout=hparams["large_angle_dropout"])
    # for i in data:
    #     i[0].save("asdf.png")
    #     exit()
        
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
        #default_root_dir=os.path.join(results_path, "supervised", "loss_dist_sphere_fix_radius", "asdf"),
        #auto_select_gpus=True,
        #enable_pl_optimizer=True,
    )
    trainer.fit(pl_module)