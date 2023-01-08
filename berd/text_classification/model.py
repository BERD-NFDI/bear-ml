"""Module class for a Pytorch Lightning classifier."""

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.functional import accuracy, auroc
from transformers import (
    AdamW,
    BertModel,
    get_linear_schedule_with_warmup,
)


class ToxicCommentTagger(pl.LightningModule):
    """Model module for classification."""

    def __init__(
        self, n_classes: int, n_training_steps=None, n_warmup_steps=None, label_col=None
    ) -> None:
        """
        Initialize a ClassifierModule.

        :param label_col: Labels .
        :param n_classes: Number of classes in dataset.
        :param n_training_steps: Number of steps training steps for scheduler .
        :param n_warmup_steps: Number of warm up steps for scheduler.
        """
        super().__init__()
        self.LABEL_COLUMNS = label_col
        # To create our model, we use the Bert pretrained model from huggingface.
        self.bert = BertModel.from_pretrained('bert-base-cased', return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        # These parameters are needed for the optimizer, which is instantiated
        # in the `configure_optimizers` method.
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps

        self.criterion = nn.BCELoss()
        self.bert.eval()
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass of the model."""
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        """Do a training step."""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return {'loss': loss, 'predictions': outputs, 'labels': labels}

    def training_epoch_end(self, outputs):
        """Do a training epoch."""
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output['labels'].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output['predictions'].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)

        for i, name in enumerate(self.LABEL_COLUMNS):
            class_roc_auc = auroc(predictions[:, i], labels[:, i])
            self.logger.experiment.add_scalar(
                f'{name}_roc_auc/Train', class_roc_auc, self.current_epoch
            )

    def validation_step(self, batch, batch_idx):
        """Do a validation step."""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Do a testing step."""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log('test_loss', loss, prog_bar=True, logger=True)
        return {'loss': loss, 'predictions': outputs, 'labels': labels}

    def testing_epoch_end(self, outputs):
        """Do a testing epoch."""
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output['labels'].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output['predictions'].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)

        acc = accuracy(predictions, labels, threshold=0.5)
        self.log('Test_accuracy', acc, prog_bar=True, logger=True)

    def configure_optimizers(self):
        """Initialize optimizer."""
        optimizer = AdamW(self.parameters(), lr=2e-5)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps,
        )

        return dict(
            optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval='step')
        )
