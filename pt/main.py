from dataset import AutoDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pl_module import AutoModule
import os
from torchvision.transforms import transforms

hparams = {"learning_rate": 1e-3, "batch_size": 64, "weight_decay": 1e-5, "img_size": 256}
std = 1
mean = 0

transforms_compose = transforms.Compose([
        transforms.Resize(hparams["img_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

folders = [f"/share/user{i}" for i in range(6, 23)]

folders = list(filter(lambda p: os.path.exists(p), folders))

if __name__ == "__main__":


    data = AutoDataset(folders, transforms_compose)
    pl_module = AutoModule(hparams, data)


    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        strict=True,
        verbose=True,
        mode='min'
    )
    model_checkpoint = ModelCheckpoint(
                monitor='val_loss',
                save_last=True,
                save_top_k=1,
    )
    callbacks = [early_stop_callback, model_checkpoint]

    trainer = pl.Trainer(
        #row_log_interval=1,
        #track_grad_norm=2,
        # weights_summary=None,
        #distributed_backend='dp',
        callbacks=callbacks,
        max_epochs=50,
        deterministic=True,
        #profiler=True,
        #fast_dev_run=True,
        gpus=1, #[0, 1],
        #default_root_dir=os.path.join(results_path, "supervised", "loss_dist_sphere_fix_radius", "asdf"),
        #auto_select_gpus=True,
        #enable_pl_optimizer=True,
    )
    trainer.fit(pl_module)
