"""Datamodule for ToxicComment dataset."""

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast as BertTokenizer


class ToxicCommentDataModule(pl.LightningDataModule):
    """ToxicComment lightning data module."""

    def __init__(
        self, train_df, test_df, tokenizer, batch_size=8, max_token_len=128
    ) -> None:
        """
        Initialize a ToxicCommentDataModule.

        :param train_df: Training data.
        :param test_df: Testing data.
        :param tokenizer: Name for tokenizer model.
        :param batch_size: Batch size.
        :param max_token_len: Maximum length of tokens .
        """
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def setup(self, stage=None):
        """
        Initialize datasets.

        This function is called by every process in a multi GPU setting and is
        dependent on the `stage` in which the lightning trainer is currently in.
        Assignments should be made here. The function serves well e.g. for
        train/test/val splits or initializing datasets.
        """
        self.train_dataset = ToxicCommentsDataset(
            self.train_df, self.tokenizer, self.max_token_len
        )

        self.test_dataset = ToxicCommentsDataset(
            self.test_df, self.tokenizer, self.max_token_len
        )

    def train_dataloader(self):
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )

    def val_dataloader(self):
        """Get validation dataloader."""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):
        """Get testing dataloader."""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=2)


class ToxicCommentsDataset(Dataset):
    """This class is a simple dataset module to prepare Toxic comment dataset after tokenization."""

    def __init__(
        self, data: pd.DataFrame, tokenizer: BertTokenizer, max_token_len: int = 128
    ):
        """Initialize ToxicComment dataset."""
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len

    def __len__(self):
        """Get length of dataset."""
        return len(self.data)

    def __getitem__(self, index: int):
        """Return an element of the dataset."""
        data_row = self.data.iloc[index]
        LABEL_COLUMNS = self.data.columns.tolist()[2:]
        c_text = data_row.comment_text
        labels = data_row[LABEL_COLUMNS]
        enc = self.tokenizer.encode_plus(
            c_text,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return dict(
            comment_text=c_text,
            input_ids=enc['input_ids'].flatten(),
            attention_mask=enc['attention_mask'].flatten(),
            labels=torch.FloatTensor(labels),
        )


""""""
