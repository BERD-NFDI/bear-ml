"""
Script for training a image segmentation with Pytorch Lightning.

This code is adapted based on
https://github.com/qubvel/segmentation_models.pytorch.
"""  # noqa
from typing import Dict, Tuple

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from torch import Tensor
from torchmetrics import JaccardIndex
from torchvision.transforms import Normalize


class SegmentationModel(pl.LightningModule):
    """Model module for segmentation."""

    def __init__(
        self,
        arch: str = 'FPN',
        encoder_name: str = 'resnet34',
        out_classes: int = 1,
        lr: float = 1e-3,
        wd: float = 0,
        freeze_encoder: bool = False,
        freeze_head: bool = False,
        freeze_decoder: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize a ClassifierModule.

        :param encoder_name: Choice of encoder model from segmentation models .
        :param in_channels: Number of inout channels.
        :param out_classes: Number of output classes.
        :param lr: learning rate.
        :param wd: weight decay.
        :param freeze_encoder: Whether to freeze all encoder weights (Default: False).
        :param freeze_head: Whether to freeze weights in the
        segmentation head (Default: False).
        :param freeze_decoder: Whether to Freeze weight in the decoder (Default: False).
        """
        super().__init__()

        # Import model from the segmentation_models_pytorch package
        # with pretrained weights on Imagenet.
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            classes=out_classes,
            **kwargs,
        )

        # Different part of model weights can be frozen by iterating over the
        # respective parameters and setting the `requires_grad` attribute to False.
        for param in self.model.encoder.parameters():
            if freeze_encoder:
                param.requires_grad = False

        for param in self.model.segmentation_head.parameters():
            if freeze_head:
                param.requires_grad = False

        for param in self.model.decoder.parameters():
            if freeze_decoder:
                param.requires_grad = False

        # We obtain the normalization parameters on which the model was trained.
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.normalize_func = Normalize(mean=params['mean'], std=params['std'])

        # The smp package offers a variety of segmentation losses.
        # We decided to use the classic Dice loss.
        # The 'from_logits' argument implies that we will input raw class scores into
        # the loss function.
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        self.lr = lr
        self.wd = wd

        # We track the Intersection-over-Union (Jaccard) metric during training.
        task = 'binary' if out_classes == 1 else 'multiclass'
        self.train_iou = JaccardIndex(task, num_classes=out_classes)
        self.val_iou = JaccardIndex(task, num_classes=out_classes)
        self.test_iou = JaccardIndex(task, num_classes=out_classes)

    def forward(self, image: Tensor) -> Tensor:
        """Forward pass of the model."""
        image = image.float() / 255
        image = self.normalize_func(image)
        return self.model(image)

    def shared_step(self, batch: Dict) -> Tuple[Tensor, Tensor]:
        """Do actions that are shared among all stages."""
        image = batch['image']
        mask = batch['mask']

        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)
        return loss, logits_mask

    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        """Do a training step."""
        loss, prediction = self.shared_step(batch)

        self.train_iou(prediction, batch['mask'])
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/iou', self.train_iou, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Do a validation step."""
        loss, prediction = self.shared_step(batch)

        self.val_iou(prediction, batch['mask'])
        self.log('val/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val/iou', self.val_iou, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Do a testing step."""
        loss, prediction = self.shared_step(batch)

        self.test_iou(prediction, batch['mask'])
        self.log('test/loss', loss, on_step=True, on_epoch=True)
        self.log('test/iou', self.test_iou, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self) -> Dict:
        """Initialize optimizer."""
        # If in doubt which optimizer to use: Adam is a reasonable choice and well
        # performing choice.
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.wd
        )
        # Learning rate schedulers can be used to modify the learning rate during
        # training. Here, we showcase Cosine Annealing, which reduces the learning
        # rate on a cosine schedule until the `T_max` epoch is reached. Then, the
        # process is repeated.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }
