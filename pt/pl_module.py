import copy
import os
from typing import Dict, Optional, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch import optim
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms
from functools import partial
from pytorch_lightning.metrics import Metric
from dataset import AutoDataset
from model import SimpleModel

DEBUG = False
MULTI_GPU = False
NUM_WORKERS = 0 #os.cpu_count() if not DEBUG else 0
SPLIT = (0.6, 0.2, 0.2)




class AutoModule(pl.LightningModule):

    def __init__(self, hparams: Dict, data: AutoDataset=None):
        super().__init__()
        self.hparams.update(hparams)
        self.model = SimpleModel()
        if data is not None:
            train_len = int(SPLIT[0]*len(data))
            val_len = int(SPLIT[1]*len(data))
            test_len = len(data)- train_len - val_len
            data = random_split(data, (train_len, val_len, test_len), generator=torch.Generator().manual_seed(42))
            self.data = {"train": data[0], "val": data[1], "test": data[2]}
        self.crit = torch.nn.MSELoss()
        # self.float()
        self.save_hyperparameters()


    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        x, y, velo = batch
        y_hat = self.model(x)
        loss = self.crit(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, velo = batch
        y_hat = self.model(x)
        loss = self.crit(y_hat, y)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y, velo = batch
        y_hat = self.model(x)
        loss = self.crit(y_hat, y)
        self.log("val_loss", loss)


    def train_dataloader(self):
        return DataLoader(self.data["train"], batch_size=self.hparams["batch_size"], num_workers=NUM_WORKERS,
                              drop_last=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.data["val"], batch_size=self.hparams["batch_size"], num_workers=NUM_WORKERS,
                              drop_last=True, pin_memory=True)


    def test_dataloader(self):
        return DataLoader(self.data["test"], batch_size=self.hparams["batch_size"], num_workers=NUM_WORKERS,
                              drop_last=True, pin_memory=True)

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), self.hparams['learning_rate'], weight_decay=self.hparams.get('weight_decay', 0))
