"""
Script for training a classifier on ToxicComment dataset  with Pytorch Lightning.

This code is adapted based on
https://curiousily.com/posts/multi-label-text-classification-with-bert-and-pytorch-lightning/
"""  # noqa


import argparse
import os
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import BertTokenizerFast as BertTokenizer

from bearml.text_classification.data import ToxicCommentDataModule
from bearml.text_classification.model import ToxicCommentTagger

RANDOM_SEED = 42


def parse_option():
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bert_model_name', type=str, default='bert-base-cased', help='bert model'
    )
    parser.add_argument(
        '--max_token_count', type=int, default=512, help='maximum number of tokens'
    )
    parser.add_argument('--batch_size', type=int, default=12, help='batch_size')
    parser.add_argument(
        '--epochs', type=int, default=10, help='number of training epochs'
    )
    parser.add_argument(
        '--data_dir', type=str, default='data', help='path to custom dataset'
    )
    parser.add_argument(
        '--log_dir', type=str, default='logs', help='path to data directory'
    )
    parser.add_argument('--n_workers', type=int, default=8, help='number of workers')

    args = parser.parse_args()
    return args


def main() -> None:
    """Run training."""
    args = parse_option()

    # PyTorch Lightning allows to set all necessary seeds in one function call.
    pl.seed_everything(RANDOM_SEED)

    # Initialize data module
    # ----------------------------------------------------------------------------------
    # Choose tokenizer from huggingface`s pretrained tokenizers.
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_name)
    data_module = ToxicCommentDataModule(
        tokenizer=tokenizer,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        max_token_len=args.max_token_count,
    )

    # Initialize model module
    # ----------------------------------------------------------------------------------

    # Exceptionally we call the preparation steps of the data module manually,
    # as we would like to access the length of the training dataset.
    # Otherwise, the dataset attribute would be set to 'None'.
    data_module.prepare_data()
    data_module.setup(stage='fit')

    # Obtain number of steps for learning rate scheduler
    steps_per_epoch = len(data_module.train_dataset) // args.batch_size  # type: ignore
    total_training_steps = steps_per_epoch * args.epochs
    warmup_steps = total_training_steps // 5

    model = ToxicCommentTagger(
        bert_model_name=args.bert_model_name,
        n_classes=data_module.train_dataset.num_classes,  # type: ignore
        label_columns=data_module.train_dataset.label_columns,  # type: ignore
        n_warmup_steps=warmup_steps,
        n_training_steps=total_training_steps,
    )

    # Setup trainer
    # ----------------------------------------------------------------------------------
    run_id = datetime.now().strftime('toxic_comments_class_%Y_%m_%d_%H_%M')
    log_path = os.path.join(args.log_dir, run_id)
    os.makedirs(log_path, exist_ok=True)

    # Define Callbacks
    # This callback always keeps a checkpoint of the best model according to
    # validation loss.
    checkpoint_callback = ModelCheckpoint(
        dirpath=log_path,
        filename='checkpoint',
        save_top_k=1,
        verbose=True,
        monitor='val/loss',
        mode='min',
    )
    # The tensorboard logger allows for monitoring the progress of training
    logger = TensorBoardLogger(save_dir=log_path)

    # Early stopping interrupts training, if there was no improvement in validation
    # loss for a certain training period.
    early_stopping_callback = EarlyStopping(monitor='val/loss', patience=2)

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
            # The progress bar will be refreshed only every 30 steps.
            TQDMProgressBar(refresh_rate=30),
        ],
        max_epochs=args.epochs,
        accelerator='cuda',
    )
    # Finally, kick of the training process.
    trainer.fit(model, data_module)

    # Evaluate how good our model works on the test data.
    trainer.test(model, dataloaders=data_module.test_dataloader())


if __name__ == '__main__':
    main()
