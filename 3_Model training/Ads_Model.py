import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pack_sequence,
)
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy
from argparse import ArgumentParser, Namespace

# from Ads_Dataloader import AdsDataModule


class AdsLSTM(LightningModule):
    def __init__(self, hparams: Namespace):
        super(AdsLSTM, self).__init__()
        self.save_hyperparameters(hparams)
        self.num_layers = self.hparams.num_layers
        self.dropout = self.hparams.dropout
        self.hidden_size = self.hparams.hidden_size
        self.lstm = nn.LSTM(
            self.hparams.input_size,
            self.hparams.hidden_size,
            self.hparams.num_layers,
            batch_first=False,
            dropout=self.dropout,
            bidirectional=True,
        )
        self.fc = nn.Linear(self.hparams.hidden_size, self.hparams.num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, X: pack_sequence):
        h0 = torch.zeros(
            self.hparams.num_layers * 2, len(X.sorted_indices), self.hparams.hidden_size
        )
        c0 = torch.zeros(
            self.hparams.num_layers * 2, len(X.sorted_indices), self.hparams.hidden_size
        )
        out, (hn, cn) = self.lstm(
            X,
            (
                h0.to(self.lstm.weight_hh_l0.device),
                c0.to(self.lstm.weight_hh_l0.device),
            ),
        )
        out = self.fc(hn[-1])
        return out

    def training_step(self, batch, i):
        # for i, batch in enumerate(AdsDataModule.train_dataloader(self)):
        (X, X_len), y_train = batch
        X_packed = pack_padded_sequence(
            X, X_len, batch_first=True, enforce_sorted=False
        )
        # y_train = label_encoder.transform(y_train)
        # y_train = torch.from_numpy(y_train)

        # Forward pass
        outputs = self(X_packed)
        loss = self.criterion(outputs, y_train)
        _, y_predicted_train = torch.max(outputs.data, 1)

        acc = accuracy(y_train, y_predicted_train)
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch, i):

        (X, X_len), y_test = batch
        X_packed = pack_padded_sequence(
            X, X_len, batch_first=True, enforce_sorted=False
        )

        # Forward pass
        outputs = self(X_packed)
        val_loss = self.criterion(outputs, y_test)
        _, y_predicted_test = torch.max(outputs.data, 1)

        val_acc = accuracy(y_test, y_predicted_test)

        self.log(
            "val_acc", val_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "val_loss",
            val_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return val_loss

    def test_step(self, batch, i):
        (X, X_len), y_val = batch
        X_packed = pack_padded_sequence(
            X, X_len, batch_first=True, enforce_sorted=False
        )
        # Forward pass
        outputs = self(X_packed)
        test_loss = self.criterion(outputs, y_val)
        _, y_pred_val = torch.max(outputs.data, 1)
        test_acc = accuracy(y_val, y_pred_val)

        self.log(
            "test_acc",
            test_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test_loss",
            test_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return test_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), weight_decay=self.hparams.weight_decay
        )
        return optimizer

    @classmethod
    def add_model_specific_args(cls, parser: ArgumentParser) -> ArgumentParser:

        parser.add_argument(
            "--num_layers",
            default=1,
            type=int,
            help="number of layers",
        )

        parser.add_argument(
            "--weight_decay",
            default=0.0004032681560146661,
            type=float,
            help="weight_decay",
        )

        parser.add_argument(
            "--dropout",
            default=0.22,
            type=float,
            help="dropout rate",
        )
        parser.add_argument(
            "--hidden_size",
            default=128,
            type=list,
            help="hidden size",
        )

        parser.add_argument(
            "--num_classes",
            default=12,
            type=int,
            help="number of classes",
        )

        parser.add_argument(
            "--input_size",
            default=2048,
            type=int,
            help="input size",
        )

        return parser
