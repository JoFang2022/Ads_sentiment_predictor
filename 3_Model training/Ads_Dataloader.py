import argparse
import pickle
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning import LightningDataModule
from typing import Optional
from argparse import Namespace

"""
Create a datamodule class which inherits pytorchlightning.DataModule.
"""


class AdsDataset(Dataset):
    def __init__(self, video_labels, valid_video_files, features):
        super(AdsDataset, self).__init__()
        self.video_labels = valid_video_files["Video_Sentiments_Score"].tolist()
        self.features = valid_video_files["Video_Features"].tolist()
        self.valid_video_files = valid_video_files

    def __len__(self):
        return len(self.valid_video_files)

    def __getitem__(self, idx):
        return (torch.Tensor(self.features[idx]), self.video_labels[idx])


class AdsDataModule(LightningDataModule):
    def __init__(self, hparams: Namespace):
        super(AdsDataModule, self).__init__()
        self.save_hyperparameters(hparams)
        self.train_df = pickle.load(open(hparams.train_pkl_file, "rb"))
        self.val_df = pickle.load(open(hparams.val_pkl_file, "rb"))
        self.label_encoder = pickle.load(open(hparams.label_encoder_file, "rb"))
        self.test_df = pickle.load(open(hparams.test_pkl_file, "rb"))

    def pad_collate(self, batch):
        (X, y) = zip(*batch)
        x_lens = torch.Tensor([len(x) for x in X])
        xx_pad = pad_sequence(X, batch_first=True, padding_value=0)
        y_encoded = self.label_encoder.transform(y)
        return (xx_pad, x_lens), torch.LongTensor(y_encoded)

    def setup(self, stage: Optional[str] = None):
        self.train_video_labels = self.train_df["Video_Sentiments_Score"]
        self.train_valid_video_files = self.train_df
        self.train_features = self.train_df["Video_Features"]
        self.train_dataset = AdsDataset(
            self.train_video_labels, self.train_valid_video_files, self.train_features
        )

        self.test_video_labels = self.test_df["Video_Sentiments_Score"]
        self.test_valid_video_files = self.test_df
        self.test_features = self.test_df["Video_Features"]
        self.test_dataset = AdsDataset(
            self.test_video_labels, self.test_valid_video_files, self.test_features
        )

        self.val_video_labels = self.val_df["Video_Sentiments_Score"]
        self.val_valid_video_files = self.val_df
        self.val_features = self.val_df["Video_Features"]
        self.val_dataset = AdsDataset(
            self.val_video_labels, self.val_valid_video_files, self.val_features
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=32, collate_fn=self.pad_collate
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=32, collate_fn=self.pad_collate)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=32, collate_fn=self.pad_collate)

    def add_data_specific_args(parser):
        parser.add_argument(
            "--train_pkl_file",
            default="valid_train.pkl",
            type=str,
            help="Train pkl file with video id and annotations",
        )
        parser.add_argument(
            "--val_pkl_file",
            default="valid_val.pkl",
            type=str,
            help="Validation pkl file with video id and annotations",
        )
        parser.add_argument(
            "--test_pkl_file",
            default="valid_test.pkl",
            type=str,
            help="Test pkl file with video id and annotations",
        )
        parser.add_argument(
            "--label_encoder_file",
            default="label_encoder.pkl",
            type=str,
            help="LabelEncoder which maps original class id to restricted class ids for this experiment",
        )

        return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        add_help=True,
    )
    AdsDataModule.add_data_specific_args(parser)
    hparams = parser.parse_args()
    train_datamodule = AdsDataModule(hparams=hparams)
    train_datamodule.setup()
