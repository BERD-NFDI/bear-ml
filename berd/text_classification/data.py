"""Datamodule for ToxicComment dataset."""
import os
from typing import Dict, Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.utils import download_file_from_google_drive
from transformers import PreTrainedTokenizerBase


class ToxicCommentDataModule(pl.LightningDataModule):
    """ToxicComment lightning data module."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        data_dir: str = 'data',
        file_name: str = 'toxic_comments.csv',
        batch_size: int = 8,
        max_token_len: int = 128,
        num_workers: int = 2,
    ) -> None:
        """
        Initialize a ToxicCommentDataModule.

        :param tokenizer: Name for tokenizer model.
        :param data_dir: Directory where data will be saved.
        :param file_name: Filename for the downloaded file.
        :param batch_size: Batch size.
        :param max_token_len: Maximum length of tokens.
        :param num_workers: Number of workers for dataloader.
        """
        super().__init__()
        self.data_dir = data_dir
        self.file_name = file_name
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        # Number of parallel workers to load batches.
        # The value of zero results in one single threaded worker.
        self.num_workers = num_workers

        # Dummy declaration for datasets, which will be instantiated in setup()
        # Attribute definitions outside the constructor are bad style, but
        # we can't and shouldn't change lightnings base framework.
        self.train_dataset: Optional[ToxicCommentsDataset] = None
        self.test_dataset: Optional[ToxicCommentsDataset] = None
        self.val_dataset: Optional[ToxicCommentsDataset] = None

    def prepare_data(self) -> None:
        """
        Prepare data.

        This function serves for initial data preparation only. No assignments should
        be made here. In our case, we will just download the data. In a multi GPU
        setting this is called with one single CPU process only.
        """
        os.makedirs(self.data_dir, exist_ok=True)
        download_file_from_google_drive(
            file_id='1VuQ-U7TtggShMeuRSA_hzC8qGDl2LRkr',
            root=self.data_dir,
            filename=self.file_name,
        )

    def setup(self, stage: str) -> None:
        """
        Initialize datasets.

        This function is called by every process in a multi GPU setting and is
        dependent on the `stage` in which the lightning trainer is currently in.
        Assignments should be made here. The function serves well e.g. for
        train/test/val splits or initializing datasets.
        """
        # Load data from save .csv file into pandas dataframe
        df = pd.read_csv(os.path.join(self.data_dir, self.file_name))
        # Perform a simple train test split of 95/5 for showcasing.
        train_df, test_df = train_test_split(df, test_size=0.05)
        # Divide data in toxic and clean comments
        label_columns = df.columns.tolist()[2:]
        train_toxic = train_df[train_df[label_columns].sum(axis=1) > 0]
        train_clean = train_df[train_df[label_columns].sum(axis=1) == 0]
        # Divide data into train and validation set with split of 90/10
        train_toxic, val_toxic = train_test_split(df, test_size=0.1)
        # Sample from clean data to balance the dataset
        train_df = pd.concat([train_toxic, train_clean.sample(13_500)])
        val_df = pd.concat([val_toxic, train_clean.sample(15_00)])

        self.train_dataset = ToxicCommentsDataset(
            train_df, self.tokenizer, self.max_token_len
        )

        self.val_dataset = ToxicCommentsDataset(
            val_df, self.tokenizer, self.max_token_len
        )

        self.test_dataset = ToxicCommentsDataset(
            test_df, self.tokenizer, self.max_token_len
        )

    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Get testing dataloader."""
        return DataLoader(
            dataset=self.test_dataset, batch_size=self.batch_size, num_workers=2
        )


class ToxicCommentsDataset(Dataset):
    """Dataset for preparation of toxic comment dataset after tokenization."""

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_token_len: int = 128,
    ) -> None:
        """Initialize ToxicComment dataset."""
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len
        self.label_columns = self.data.columns.tolist()[2:]
        self.num_classes = len(self.label_columns)

    def __len__(self) -> int:
        """Get length of dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> Dict:
        """Return an element of the dataset."""
        data_row = self.data.iloc[index]
        c_text = data_row.comment_text
        labels = data_row[self.label_columns]

        enc = self.tokenizer.encode_plus(
            c_text,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'comment_text': c_text,
            'input_ids': enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'labels': torch.FloatTensor(labels),
        }
