#!/usr/bin/env python
# coding: utf-8
import torch.optim
import pytorch_lightning as pl


class LitTrainer(pl.LightningModule):
    def __init__(self, model, loss_fn, optim):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optim = optim

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(torch.float32)

        y_pred = self.model(x).reshape(1, -1)
        train_loss = self.loss_fn(y_pred, y)

        self.log("train_loss", train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        x = x.to(torch.float32)

        y_pred = self.model(x).reshape(1, -1)
        validate_loss = self.loss_fn(y_pred, y)

        self.log("val_loss", validate_loss)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        x = x.to(torch.float32)

        y_pred = self.model(x).reshape(1, -1)
        test_loss = self.loss_fn(y_pred, y)

        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        return self.optim
