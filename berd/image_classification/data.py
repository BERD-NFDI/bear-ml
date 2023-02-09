"""Datamodule for CIFAR10."""
import os
from typing import (
    Any,
    List,
    Optional,
    Tuple,
)

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import (
    DataLoader,
    Dataset,
    random_split,
)
from torchvision.datasets import CIFAR10
from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
)


class CIFAR10DataModule(pl.LightningDataModule):
    """CIFAR10 lightning data module."""

    def __init__(
        self,
        data_dir: str = 'data',
        batch_size: int = 8,
        num_workers: int = 4,
        transform: Optional[List] = None,
        seed: int = 42,
    ) -> None:
        """
        Initialize a CIFAR10DataModule.

        :param data_dir: Directory where CIFAR10 will be saved.
        :param batch_size: Batch size.
        :param num_workers: Number of workers for dataloader.
        :param transform: Augmentations for training.
        :param seed: Random seed.
        """
        super().__init__()
        # Directory where the downloaded dataset will be saved
        self.data_dir = data_dir
        self.batch_size = batch_size
        # Number of parallel workers to load batches.
        # The value of zero results in one single threaded worker.
        self.num_workers = num_workers
        # Seed to reproduce our random data split, which we will do later
        self.seed = seed

        # This transformation will be called by all dataloaders.
        base_transform = [
            # The data is loaded to memory as 8-bit PIL image.
            # ToTensor() converts this to a tensor, which is
            # normalized to a (0, 1) range.
            ToTensor(),
            # A standard practice is to scale the data so that it has zero mean
            # and a standard deviation of one. The below values were precomputed
            # from the training dataset (and taken from
            # https://github.com/kuangliu/pytorch-cifar/issues/19#issue-268972488).
            Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.247, 0.243, 0.261),
            ),
        ]

        self.base_transform = Compose(base_transform)
        # We allow adding more training transforms like
        # augmentations over the constructor.
        if transform is None:
            self.train_transform = self.base_transform
        else:
            self.train_transform = Compose(transform + base_transform)

        # Dummy declaration for datasets, which will be instantiated in setup()
        # Attribute definitions outside the constructor are bad style, but
        # we can't and shouldn't change lightnings base framework.
        self.train_dataset: Optional[CIFAR10Wrapper] = None
        self.val_dataset: Optional[CIFAR10Wrapper] = None
        self.test_dataset: Optional[CIFAR10] = None

    def prepare_data(self) -> None:
        """
        Prepare data.

        This function serves for initial data preparation only. No assignments should
        be made here. In our case, we will just download the data. In a multi GPU
        setting this is called with one single CPU process only.
        """
        os.makedirs(self.data_dir, exist_ok=True)
        CIFAR10(root=self.data_dir, train=True, download=True)
        CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage: str) -> None:
        """
        Initialize datasets.

        This function is called by every process in a multi GPU setting and is
        dependent on the `stage` in which the lightning trainer is currently in.
        Assignments should be made here. The function serves well e.g. for
        train/test/val splits or initializing datasets.
        """
        dataset = CIFAR10(root=self.data_dir, train=True)

        # The stage-parameter corresponds to the function names of the lightning trainer
        if stage == 'fit':
            # For this example case, we take 20% of our training dataset for validation.
            # Validation is done after every epoch, whereas testing is only done once
            # our training has finished to check our generalization abilities.
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(
                dataset=dataset,
                lengths=(train_size, val_size),
                generator=torch.Generator().manual_seed(self.seed),
            )

            # Thanks to our wrapper we can now set the transformations after
            # initializing the datasets. We don't want any augmentations happening
            # in our validation and test phase!
            self.train_dataset = CIFAR10Wrapper(train_dataset, self.train_transform)
            self.val_dataset = CIFAR10Wrapper(val_dataset, self.base_transform)

        if stage == 'test':
            self.test_dataset = CIFAR10(
                root=self.data_dir,
                train=False,
                transform=self.base_transform,
            )

    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        """Get testing dataloader."""
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )


class CIFAR10Wrapper(Dataset):
    """
    Wrapper for torchvisions CIFAR10 dataset.

    This class is a simple wrapper for the default CIFAR10 torchvision dataset.
    Here it serves us solely to change data transformations more conveniently.
    """

    def __init__(
        self,
        cifar: CIFAR10,
        transform: Any = None,
    ) -> None:
        """Initialize CIFAR10Wrapper dataset."""
        self.cifar = cifar
        self.transform = transform

    def __len__(self) -> int:
        """Get length of dataset."""
        return len(self.cifar)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Return an element of the dataset."""
        img, label = self.cifar[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label
