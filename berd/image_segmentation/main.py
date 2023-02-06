"""Script for training Image segmentation with Pytorch Lightning."""

import argparse
import os
from datetime import datetime

import pytorch_lightning as pl
import torch
import utils
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TQDMProgressBar,
)
from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset
from torch.utils.data import DataLoader

from berd.image_segmentation.segmentation import segModel

RANDOM_SEED = 42

# PyTorch Lightning allows to set all necessary seeds in one function call.
pl.seed_everything(RANDOM_SEED)


def parse_option():
    """Get command line arguments."""
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument(
        '--epochs', type=int, default=50, help='number of training epochs'
    )
    parser.add_argument(
        '--data_dir', type=str, default='data', help='path to custom dataset'
    )
    parser.add_argument(
        '--log_dir', type=str, default='logs', help='path to data directory'
    )

    parser.add_argument('--n_workers', type=int, default=16, help='number of workers')
    parser.add_argument(
        '--model_w_f',
        default=False,
        type=utils.bool_flag,
        help='Whether to Freeze weight in the model (Default: False)',
    )
    parser.add_argument(
        '--head_w_f',
        default=False,
        type=utils.bool_flag,
        help='Whether to Freeze weight in the segmentation head (Default: False)',
    )
    parser.add_argument(
        '--decoder_w_f',
        default=False,
        type=utils.bool_flag,
        help='Whether to Freeze weight in the decoder (Default: False)',
    )

    opt = parser.parse_args()

    # set the path according to the environment
    if opt.data_dir is None:
        opt.data_dir = '~/data'
    opt.save_folder = './saved_models/'
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    return opt


def main():
    """Run training."""
    opt = parse_option()
    # Import dataset
    SimpleOxfordPetDataset.download(opt.data_dir)
    train_dataset = SimpleOxfordPetDataset(opt.data_dir, 'train')
    valid_dataset = SimpleOxfordPetDataset(opt.data_dir, 'valid')
    test_dataset = SimpleOxfordPetDataset(opt.data_dir, 'test')

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_workers,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_workers,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_workers,
    )
    # Initialize model module
    model = segModel(
        'FPN',
        'resnet34',
        in_channels=3,
        out_classes=1,
        lr=opt.lr,
        wd=opt.d,
        model_w_f=opt.model_w_f,
        head_w_f=opt.head_w_f,
        decoder_w_f=opt.decoder_w_f,
    )
    # Define Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='./saved_models/',
        filename='best-checkpoint',
        verbose=True,
        monitor='valid_loss',
        mode='min',
    )
    # Setup trainer
    # ----------------------------------------------------------------------------------
    run_id = datetime.now().strftime('image_segmentation_%Y_%m_%d_%H_%M')
    log_path = os.path.join(opt.log_dir, run_id)
    os.makedirs(log_path, exist_ok=True)
    # Early stopping interrupts training, if there was no improvement in validation
    # loss for a certain training period.
    early_stopping_callback = EarlyStopping(monitor='valid_loss', patience=8)
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    else:
        n_gpus = 1
    # Instantiate trainer object.
    # The lightning trainer has a large number of parameters that can improve the
    # training experience. It is recommended to check out the lightning docs for
    # further information.
    # Lightning allows for simple multi-gpu training, gradient accumulation, half
    # precision training, etc. using the trainer class.
    trainer = pl.Trainer(
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
            TQDMProgressBar(refresh_rate=30),
        ],
        max_epochs=5,
        strategy='ddp_find_unused_parameters_false',
        gpus=n_gpus,
    )
    # Finally, kick of the training process.
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )
    valid_metrics = trainer.validate(model, dataloaders=valid_dataloader, verbose=False)
    print(valid_metrics)
    # Evaluate how good our model works on the test data.
    test_metrics = trainer.test(
        model,
        dataloaders=test_dataloader,
    )
    print(test_metrics)


if __name__ == '__main__':
    main()
