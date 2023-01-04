"""Module class for a Pytorch Lightning classifier."""

from typing import Dict, Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
)


class ClassifierModule(pl.LightningModule):
    """Model module for classification."""

    def __init__(
        self,
        model_id: str = 'resnet34',
        num_classes: int = 10,
        learning_rate: float = 1e-1,
        weight_decay: float = 0,
        lr_patience: int = 3,
    ) -> None:
        """
        Initialize a ClassifierModule.

        :param model_id: Torchvision model id.
        :param num_classes: Number of classes in dataset.
        :param learning_rate: Learning rate.
        :param weight_decay: L2-regularization
        :param lr_patience: Patience for learning rate scheduler.
        """
        super().__init__()
        self.num_classes = num_classes

        # To create our model, we use the torch hub, which gives  access to a variety
        # of architectures without the need to build and compose the model yourself.
        # We use the standard pytorch repository for vision models.
        # We need an ID to specify our desired model. All available models can
        # be listed by calling `torch.hub.list(repository)`.
        # We could also easily load a pretrained model by specifying the `weights`
        # parameter.
        repository = 'pytorch/vision'
        self.model = torch.hub.load(
            repository, model_id, weights=None, num_classes=num_classes
        )

        # These parameters are needed for the optimizer, which is instantiated
        # in the `configure_optimizers` method.
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_patience = lr_patience

        # The cross-entropy loss function is perfect for multi-class classification.
        self.loss_func = nn.CrossEntropyLoss()

        # Metrics are tracked using the torchmetrics package.
        # This gives a variety of convenience including sync over devices.
        # With the collection, multiple metrics can be bundled and then cloned for the
        # respective stage.
        metrics = MetricCollection(
            [
                MulticlassAccuracy(num_classes),
                MulticlassPrecision(num_classes),
                MulticlassRecall(num_classes),
            ]
        )
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model."""
        return self.model(x)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Do a training step."""
        # The batch is a tuple, which is unpacked.
        x, y = batch
        # Execute a forward pass to get predictions
        y_hat = self(x)
        # Compute loss with loss function
        loss = self.loss_func(y_hat, y)

        # The loss is logged on every step, whereas the other metrics are
        # computed only once at the end of the epoch.
        self.train_metrics(y_hat, y)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        # We don't need to do a manual backward pass and apply gradients.
        # The lightning framework does this for us when returning the loss in the
        # training step.
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Do a validation step."""
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)

        self.val_metrics(y_hat, y)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Do a testing step."""
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)

        self.test_metrics(y_hat, y)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self) -> Dict:
        """Initialize optimizer."""
        # The choice is the classic SGD, which is very powerful in combination with
        # momentum and a learning rate scheduler
        optimizer = SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=self.weight_decay,
        )
        # This scheduler reduces the learning rate, if the validation loss didn't
        # decrease in the number of epochs specified by patience.
        scheduler = ReduceLROnPlateau(
            optimizer, factor=0.1, patience=self.lr_patience, min_lr=1e-5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'frequency': 1,
            },
        }
