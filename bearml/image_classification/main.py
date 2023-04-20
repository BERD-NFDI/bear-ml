"""Script for training a classifier on CIFAR10 with Pytorch Lightning."""

import argparse
import os
from datetime import datetime

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

from bearml.image_classification.data import CIFAR10DataModule
from bearml.image_classification.model import ClassifierModule

RANDOM_SEED = 42


def get_args():
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument(
        '--epochs', type=int, default=20, help='number of training epochs'
    )
    parser.add_argument(
        '--model_id', type=str, default='resnet18', help='model id for torch hub'
    )
    parser.add_argument(
        '--data_dir', type=str, default='data', help='path to data directory'
    )
    parser.add_argument(
        '--log_dir', type=str, default='logs', help='path to data directory'
    )
    parser.add_argument('--lr', type=float, default=1e-1, help='learning rate')
    parser.add_argument('--wd', type=float, default=0, help='weight decay')
    parser.add_argument('--n_cls', type=int, default=10, help='number of classes')
    parser.add_argument('--n_workers', type=int, default=8, help='number of workers')
    args = parser.parse_args()
    return args


def main() -> None:
    """Run training."""
    args = get_args()

    # PyTorch Lightning allows to set all necessary seeds in one function call.
    pl.seed_everything(RANDOM_SEED)

    # Initialize model module
    # ----------------------------------------------------------------------------------
    model_module = ClassifierModule(
        model_id=args.model_id,
        num_classes=args.n_cls,
        learning_rate=args.lr,
        weight_decay=args.wd,
    )

    # Initialize data module
    # ----------------------------------------------------------------------------------

    # While there is a wide variety of different augmentation strategies, we simply
    # resort to the supposedly optimal AutoAugment policy.
    transform = AutoAugment(policy=AutoAugmentPolicy.CIFAR10)
    data_module = CIFAR10DataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        transform=[transform],
    )

    # Setup trainer
    # ----------------------------------------------------------------------------------
    run_id = datetime.now().strftime('cifar_class_%Y_%m_%d_%H_%M')
    log_path = os.path.join(args.log_dir, run_id)
    os.makedirs(log_path, exist_ok=True)

    # Define Callbacks
    # This callback always keeps a checkpoint of the best model according to
    # validation accuracy.
    checkpoint_callback = ModelCheckpoint(
        dirpath=log_path,
        filename='checkpoint',
        save_top_k=1,
        verbose=True,
        monitor='val/MulticlassAccuracy',
        mode='max',
    )
    # Early stopping interrupts training, if there was no improvement in validation
    # loss for a certain training period.
    early_stopping_callback = EarlyStopping(monitor='val/loss', patience=5)

    # The tensorboard logger allows for monitoring the progress of training
    logger = TensorBoardLogger(save_dir=log_path)

    # In this simple example we just check if a GPU is available.
    # For training larger models in a distributed settings, this needs more care.
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    print('Using device {}.'.format(accelerator))

    # Instantiate trainer object.
    # The lightning trainer has a large number of parameters that can improve the
    # training experience. It is recommended to check out the lightning docs for
    # further information.
    # Lightning allows for simple multi-gpu training, gradient accumulation, half
    # precision training, etc. using the trainer class.
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        max_epochs=args.epochs,
        accelerator=accelerator,
    )

    # Finally, kick of the training process.
    trainer.fit(model_module, data_module)
    # Evaluate how good our model works on the test data.
    trainer.test(model_module, datamodule=data_module)


if __name__ == '__main__':
    main()
