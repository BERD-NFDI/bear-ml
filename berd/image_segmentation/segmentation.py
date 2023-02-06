"""
Script for training a image segmentation with Pytorch Lightning.

This code is adapted based on
https://github.com/qubvel/segmentation_models.pytorch.
"""  # noqa

import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch

torch.cuda.empty_cache()


class segModel(pl.LightningModule):
    """Model module for segmentation."""

    def __init__(
        self,
        arch,
        encoder_name,
        in_channels,
        out_classes,
        lr,
        wd,
        model_w_f,
        head_w_f,
        decoder_w_f,
        **kwargs,
    ):
        """
        Initialize a ClassifierModule.

        :param encoder_name: Choice of encoder model from pytorch segmentation .
        :param in_channels: Number of inout channels.
        :param out_classes: Number of output classes.
        :param lr: learning rate.
        :param wd: weight decay.
        :param model_w_f: "Whether to Freeze weight in the model (Default: False)".
        :param head_w_f: "Whether to Freeze weight in the head (Default: False)".
        :param decoder_w_f: "Whether to Freeze weight in the decoder (Default: False)".
        """
        super().__init__()
        # Import model from pytorch segmentation with pretrained weights on Imagenet
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )
        # Here different part of model weights can be froze
        for param in self.model.parameters():
            if model_w_f:
                param.requires_grad = False
            else:
                param.requires_grad = True
        for param in self.model.segmentation_head.parameters():
            if head_w_f:
                param.requires_grad = False
            else:
                param.requires_grad = True
        for param in self.model.decoder.parameters():
            if decoder_w_f:
                param.requires_grad = False
            else:
                param.requires_grad = True
        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer('std', torch.tensor(params['std']).view(1, 3, 1, 1))
        self.register_buffer('mean', torch.tensor(params['mean']).view(1, 3, 1, 1))
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.lr = lr
        self.wd = wd

    def forward(self, image):
        """Forward pass of the model."""
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        """Do a shared step."""
        image = batch['image']
        mask = batch['mask']

        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode='binary'
        )
        self.log('loss', loss)
        return {
            'loss': loss,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
        }

    def shared_epoch_end(self, outputs, stage):
        """Do shared actions after each epoch."""
        # aggregate step metics
        tp = torch.cat([x['tp'] for x in outputs])
        fp = torch.cat([x['fp'] for x in outputs])
        fn = torch.cat([x['fn'] for x in outputs])
        tn = torch.cat([x['tn'] for x in outputs])
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction='micro-imagewise'
        )
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro')
        loss = np.mean(torch.stack([x['loss'] for x in outputs], dim=0).cpu().numpy())
        metrics = {
            f'{stage}_per_image_iou': per_image_iou,
            f'{stage}_dataset_iou': dataset_iou,
            f'{stage}_loss': loss,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        """Do a training step."""
        return self.shared_step(batch, 'train')

    def training_epoch_end(self, outputs):
        """Do actions after training epoch."""
        return self.shared_epoch_end(outputs, 'train')

    def validation_step(self, batch, batch_idx):
        """Do a validation step."""
        return self.shared_step(batch, 'valid')

    def validation_epoch_end(self, outputs):
        """Do actions after validation epoch."""
        return self.shared_epoch_end(outputs, 'valid')

    def test_step(self, batch, batch_idx):
        """Do a testing step."""
        return self.shared_step(batch, 'test')

    def test_epoch_end(self, outputs):
        """Do actions after testing epoch."""
        return self.shared_epoch_end(outputs, 'test')

    def configure_optimizers(self):
        """Initialize optimizer."""
        # There are different options for optimizer if you dont have
        # enough information Adam is reasonable choice
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.wd
        )
        # There are different options for scheduler for learning rate on pytorch
        # if you dont have enough information Cosine is reasonable choice
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }
