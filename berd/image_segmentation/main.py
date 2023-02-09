"""Script for training an image segmentation model with Pytorch Lightning."""

import argparse
import os
from datetime import datetime

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset
from torch.utils.data import DataLoader

from berd.image_segmentation.model import SegmentationModel

RANDOM_SEED = 42


def parse_option():
    """Get command line arguments."""
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument(
        '--epochs', type=int, default=20, help='number of training epochs'
    )
    parser.add_argument(
        '--data_dir', type=str, default='data', help='path to custom dataset'
    )
    parser.add_argument(
        '--log_dir', type=str, default='logs', help='path to data directory'
    )
    parser.add_argument('--n_workers', type=int, default=4, help='number of workers')
    parser.add_argument(
        '--freeze_encoder',
        action='store_true',
        help='Whether to freeze all encoder weights (Default: False)',
    )
    parser.add_argument(
        '--freeze_head',
        action='store_true',
        help='Whether to freeze weights in the segmentation head (Default: False)',
    )
    parser.add_argument(
        '--freeze_decoder',
        action='store_true',
        help='Whether to Freeze weight in the decoder (Default: False)',
    )

    args = parser.parse_args()
    return args


def main():
    """Run training."""
    args = parse_option()

    # PyTorch Lightning allows to set all necessary seeds in one function call.
    pl.seed_everything(RANDOM_SEED)

    # Initialize data
    # ----------------------------------------------------------------------------------

    # For this example we use the Oxford pet dataset, which has around 200 images for
    # each of its 37 cats & dogs species.
    # An API to the dataset is provided over the segmentation_models_pytorch package.
    print('Downloading dataset ...')
    dataset_path = os.path.join(args.data_dir, 'oxford_pet')
    os.makedirs(dataset_path, exist_ok=True)

    # Quick dirty hack to check whether data was already completely downloaded.
    if len(os.listdir(os.path.join(dataset_path, 'images'))) != 7393:
        SimpleOxfordPetDataset.download(args.data_dir)
    else:
        print('Dataset is already downloaded.')

    train_dataset = SimpleOxfordPetDataset(root=dataset_path, mode='train')
    valid_dataset = SimpleOxfordPetDataset(root=dataset_path, mode='valid')
    test_dataset = SimpleOxfordPetDataset(root=dataset_path, mode='test')

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
    )

    # Initialize model module
    # ----------------------------------------------------------------------------------

    model = SegmentationModel(
        arch='FPN',
        encoder_name='resnet34',
        out_classes=1,
        lr=args.lr,
        wd=args.wd,
        freeze_encoder=args.freeze_encoder,
        freeze_head=args.freeze_head,
        freeze_decoder=args.freeze_decoder,
    )

    # Setup trainer
    # ----------------------------------------------------------------------------------
    run_id = datetime.now().strftime('image_segmentation_%Y_%m_%d_%H_%M')
    log_path = os.path.join(args.log_dir, run_id)
    os.makedirs(log_path, exist_ok=True)

    if torch.cuda.is_available():
        acc_dict = {'devices': 1, 'accelerator': 'gpu'}
    else:
        acc_dict = {'accelerator': 'cpu'}

    # The tensorboard logger allows for monitoring the progress of training.
    logger = TensorBoardLogger(save_dir=log_path)

    # Define Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=log_path,
        filename='checkpoint',
        save_top_k=1,
        verbose=True,
        monitor='val/loss',
        mode='min',
    )

    # Early stopping interrupts training, if there was no improvement in validation
    # loss for a certain training period.
    early_stopping_callback = EarlyStopping(monitor='val/loss', patience=5)

    # Instantiate trainer object.
    # The lightning trainer has a large number of parameters that can improve the
    # training experience. It is recommended to check out the lightning docs for
    # further information.
    # Lightning allows for simple multi-gpu training, gradient accumulation, half
    # precision training, etc. using the trainer class.
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
            TQDMProgressBar(refresh_rate=30),
        ],
        max_epochs=5,
        **acc_dict
    )
    # Finally, kick of the training process.
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )

    # Check how well we perform on the validation set.
    valid_metrics = trainer.validate(model, dataloaders=valid_dataloader, verbose=False)
    print('Validation Metrics:', valid_metrics)

    # Evaluate how good our model works on the test data.
    trainer.test(
        model,
        dataloaders=test_dataloader,
    )


if __name__ == '__main__':
    main()
