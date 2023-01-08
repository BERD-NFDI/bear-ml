"""Script for training a classifier on Toxicomment dataset  with Pytorch Lightning."""


import argparse
import os

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torchvision.datasets.utils import download_file_from_google_drive
from transformers import BertTokenizerFast as BertTokenizer

from berd.text_classification.data.DataModule import ToxicCommentDataModule
from berd.text_classification.model.Toxitcomment_Pl_module import ToxicCommentTagger

RANDOM_SEED = 42

pl.seed_everything(RANDOM_SEED)


def parse_option():
    """Get command line arguments."""
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument(
        '--BERT_MODEL_NAME', type=str, default='bert-base-cased', help='bert model'
    )
    parser.add_argument(
        '--MAX_TOKEN_COUNT', type=int, default=512, help='maximum number of tokens'
    )
    parser.add_argument('--batch_size', type=int, default=12, help='batch_size')
    parser.add_argument(
        '--epochs', type=int, default=10, help='number of training epochs'
    )
    parser.add_argument(
        '--data_folder', type=str, default='data', help='path to custom dataset'
    )
    parser.add_argument(
        '--log_dir', type=str, default='logs', help='path to data directory'
    )
    opt = parser.parse_args()

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './saved_models/{}_models'.format(opt.dataset)
    opt.save_folder = opt.model_path
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    return opt


def main():
    """Run training."""
    opt = parse_option()
    # import dataset
    download_file_from_google_drive(
        file_id='1VuQ-U7TtggShMeuRSA_hzC8qGDl2LRkr',
        root='opt.data_folder ',
        filename='toxic_comments.csv',
    )
    # Change  dataset format to data-frame
    df = pd.read_csv(os.path.join(opt.data_folder, '/toxic_comments.csv'))
    # Split dataset to training and testing dataset
    train_df, val_df = train_test_split(df, test_size=0.05)
    LABEL_COLUMNS = df.columns.tolist()[2:]
    # Divide data in toxic and clean comments
    train_toxic = train_df[train_df[LABEL_COLUMNS].sum(axis=1) > 0]
    train_clean = train_df[train_df[LABEL_COLUMNS].sum(axis=1) == 0]
    # Sample from clean data to balance the dataset
    train_df = pd.concat([train_toxic, train_clean.sample(15_000)])
    # Choose tokenizer form pretrained tokenizer from Huggingface.
    tokenizer = BertTokenizer.from_pretrained(opt.BERT_MODEL_NAME)
    # Initialize data module
    data_module = ToxicCommentDataModule(
        train_df,
        val_df,
        tokenizer,
        batch_size=opt.batch_size,
        max_token_len=opt.MAX_TOKEN_COUNT,
    )

    # Adjust inputs for scheduler
    steps_per_epoch = len(train_df) // opt.batch_size
    total_training_steps = steps_per_epoch * opt.epochs
    warmup_steps = total_training_steps // 5

    # Initialize model module
    model = ToxicCommentTagger(
        n_classes=len(LABEL_COLUMNS),
        label_col=LABEL_COLUMNS,
        n_warmup_steps=warmup_steps,
        n_training_steps=total_training_steps,
    )

    # Define Callbacks
    # This callback always keeps a checkpoint of the best model according to
    # validation loss.
    checkpoint_callback = ModelCheckpoint(
        dirpath=opt.model_path,
        filename='best-checkpoint',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )
    # The tensorboard logger allows for monitoring the progress of training
    logger = TensorBoardLogger(
        'lightning_logs', name='toxic-comments', save_dir=opt.log_dir
    )
    # Early stopping interrupts training, if there was no improvement in validation
    # loss for a certain training period.
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)

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
        max_epochs=opt.epochs,
        gpus=1,
    )
    # Finally, kick of the training process.
    trainer.fit(model, data_module)

    # Evaluate how good our model works on the test data.
    trainer.test(model, dataloaders=data_module.test_dataloader())


if __name__ == '__main__':
    main()
