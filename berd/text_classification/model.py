"""Module class for a Pytorch Lightning classifier."""

from typing import (
    Any,
    Dict,
    List,
)

import pytorch_lightning as pl
from torch import Tensor, nn
from torch.optim import AdamW
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelAUROC,
    MultilabelPrecision,
    MultilabelRecall,
)
from transformers import BertModel, get_linear_schedule_with_warmup

"""This code is adapted based on
https://curiousily.com/posts/multi-label-text-classification-with-bert-and-pytorch-lightning/"""


class ToxicCommentTagger(pl.LightningModule):
    """Model module for classification."""

    def __init__(
        self,
        n_classes: int,
        n_training_steps: int,
        n_warmup_steps: int,
        label_columns: List,
        bert_model_name: str = 'bert-base-cased',
    ) -> None:
        """
        Initialize a ClassifierModule.

        :param label_columns: Labels .
        :param n_classes: Number of classes in dataset.
        :param n_training_steps: Number of steps training steps for scheduler.
        :param n_warmup_steps: Number of warm up steps for scheduler.
        """
        super().__init__()
        self.label_columns = label_columns
        self.n_classes = n_classes
        # To create our model, we use the Bert pretrained model from huggingface.
        # The bert model provides embeddings for the text.
        self.bert = BertModel.from_pretrained(bert_model_name, return_dict=True)
        # The classifier is a simple linear model.
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

        # These parameters are needed for the optimizer, which is instantiated
        # in the `configure_optimizers` method.
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps

        self.loss_func = nn.BCEWithLogitsLoss()
        # To get best results you should fine-tune the bert model as well. but if you have limited computation power
        # you need to freeze the layers inside the Bert model. mode.eval() disable layer such as dropout and batch
        # normalization. By putting param.requires_grad to false, we deactivate the back-prop and improve memory usage
        self.bert.eval()
        for param in self.bert.parameters():
            param.requires_grad = False

        # Metrics are tracked using the torchmetrics package.
        # This gives a variety of convenience including sync over devices.
        # With the collection, multiple metrics can be bundled and then cloned for the
        # respective stage.
        metrics = MetricCollection(
            [
                MultilabelAccuracy(n_classes),
                MultilabelPrecision(n_classes),
                MultilabelRecall(n_classes),
            ]
        )
        self.train_metrics = metrics.clone(prefix='train/')
        self.val_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')

        self.train_auroc = MultilabelAUROC(n_classes, average='none')
        self.val_auroc = MultilabelAUROC(n_classes, average='none')
        self.test_auroc = MultilabelAUROC(n_classes, average='none')

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Forward pass of the model."""
        embeddings = self.bert(input_ids, attention_mask=attention_mask)
        return self.classifier(embeddings.pooler_output)

    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        """Do a training step."""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        predictions = self(input_ids, attention_mask)
        loss = self.loss_func(predictions, labels)

        self.train_metrics(predictions, labels)
        self.train_auroc.update(predictions, labels)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def training_epoch_end(self, outputs: Any) -> None:
        """Do actions after training epoch."""
        aurocs = self.train_auroc.compute()
        self.train_auroc.reset()

        self.log_dict(
            {'train/auroc_' + k: v for k, v in zip(self.label_columns, aurocs)}
        )

    def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
        """Do a validation step."""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        predictions = self(input_ids, attention_mask)
        loss = self.loss_func(predictions, labels)

        self.val_metrics(predictions, labels)
        self.val_auroc.update(predictions, labels)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)
        self.log('val/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_epoch_end(self, outputs: Any) -> None:
        """Do actions after validation epoch."""
        aurocs = self.val_auroc.compute()
        self.val_auroc.reset()

        self.log_dict({'val/auroc_' + k: v for k, v in zip(self.label_columns, aurocs)})

    def test_step(self, batch: Dict, batch_idx: int) -> Tensor:
        """Do a testing step."""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        predictions = self(input_ids, attention_mask)
        loss = self.loss_func(predictions, labels)

        self.test_metrics(predictions, labels)
        self.test_auroc.update(predictions, labels)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)
        self.log('test/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def testing_epoch_end(self, outputs: Any) -> None:
        """Do actions after testing epoch."""
        aurocs = self.test_auroc.compute()
        self.test_auroc.reset()

        self.log_dict(
            {'test/auroc_' + k: v for k, v in zip(self.label_columns, aurocs)}
        )

    def configure_optimizers(self) -> Dict:
        """Initialize optimizer."""
        # AdamW is Adam with weight regularization to improve the generalization power
        optimizer = AdamW(self.parameters(), lr=2e-5)
        # Decrease learning rate linearly to 0 after warmup steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            },
        }
