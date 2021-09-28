from dataset import AutoDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pl_module import AutoModule
import os
from torchvision.transforms import transforms

hparams = {"learning_rate": 1e-4, "batch_size": 32, "weight_decay": 1e-5, "img_size": 256, "workers": 4}
gpu = 0

std = 1
mean = 0

class LambdaTrans():
    def __init__(self, f):
        self.f = f

    def __call__(self, x):
        return self.f(x)


transforms_compose = transforms.Compose([
        transforms.Resize((hparams["img_size"], hparams["img_size"])), # todo: scaling!
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
        # LambdaTrans(lambda x: x/255)
    ])

folders = [f"/share/user{i}" for i in range(6, 23)]
folders = ["/home/tobi/fa/imgs"]

folders = list(filter(lambda p: os.path.exists(p), folders))

if __name__ == "__main__":


    data = AutoDataset(folders, transforms_compose)
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
        max_epochs=20,
        deterministic=True,
        #profiler=True,
        #fast_dev_run=True,
        gpus=[gpu], #[0, 1],
        #default_root_dir=os.path.join(results_path, "supervised", "loss_dist_sphere_fix_radius", "asdf"),
        #auto_select_gpus=True,
        #enable_pl_optimizer=True,
    )
    trainer.fit(pl_module)
