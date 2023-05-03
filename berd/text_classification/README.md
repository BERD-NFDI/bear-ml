# Text Classification with PyTorch Lightning

This repository is for the BERD project. It contains the example codes for text Classification with PyTorch Lightning.

## Table of contents
* [Background](#background)
* [Dataset](#dataset)
* [Code Structure](#code-structure)
* [Acknowledgements](#Acknowledgements)

## Background

### Text Classification

Text classification is the task of automatically categorizing a given text document into one or more predefined
categories. The process of text classification involves two main stages: feature extraction and model training. In the
feature extraction stage, the text data is preprocessed to extract relevant information that can be used as input to a
machine learning algorithm. This can involve techniques such as tokenization, stopword removal, stemming, and feature
selection. In the model training stage, a machine learning algorithm is trained on the preprocessed text data and
associated labels to learn a model that can classify new text instances. Popular machine learning algorithms for text
classification include Naive Bayes, Support Vector Machines (SVMs), and Neural Networks. Once the model is trained,
it can be used to predict the category of new text instances based on their features. The accuracy of the model can be
evaluated using various metrics such as precision, recall, and F1-score.


### Deep Learning

Deep learning is a subset of machine learning that uses artificial neural networks to model and solve complex problems.
Deep learning models are designed to automatically learn hierarchical representations of data by processing it through
a series of non-linear transformations.

### Neural Network Model

### Optimization

Optimization in deep learning refers to the process of the weights and biases of the neural network so that it can
accurately predict the target variable.

n order to optimize a deep learning model, an algorithm is used to iteratively update the model's parameters based on
the gradients of the loss function with respect to those parameters. This is typically done using stochastic gradient
descent (SGD) or one of its variants, such as Adam, Adagrad, or RMSProp.

The optimization process involves choosing appropriate hyperparameters, such as the learning rate and batch size, that
control the rate of parameter updates and the size of the mini-batches used during training. It also involves monitoring
the training and validation loss to prevent overfitting, which occurs when the model becomes too complex and starts to
memorize the training data instead of learning general patterns that can be applied to new data.

### Scheduler

A learning rate scheduler is a technique used in deep learning to dynamically adjust the learning rate during training
in order to improve model performance. The learning rate is a hyperparameter that controls the step size of the updates
to the model weights during optimization.

In a learning rate scheduler, the learning rate is adjusted based on a predefined schedule, which can be based on
various criteria such as the number of epochs, the validation loss, or the accuracy of the model on the training data.
The goal of a learning rate scheduler is to improve the convergence of the model by decreasing the learning rate as the
training progresses, which allows the model to make smaller updates to the weights and converge more slowly to a better
optimum.

Common types of learning rate schedules include step decay, where the learning rate is reduced by a fixed factor after
a fixed number of epochs, and exponential decay, where the learning rate is decreased exponentially over time. More
advanced learning rate schedules include cyclical learning rates, where the learning rate is cyclically increased and
decreased over time, and cosine annealing, where the learning rate is gradually reduced over a cosine-shaped curve.

### PyTorch Lightning

PyTorch Lightning is a lightweight PyTorch wrapper that simplifies the training process for complex deep learning models.
It is designed to make PyTorch code more modular, maintainable, and scalable, and to allow researchers and practitioners
to focus on the high-level aspects of their models, rather than the low-level details of the training process.

PyTorch Lightning provides a standardized interface for common training tasks, such as data loading, training loops,
and model checkpointing, which reduces the amount of boilerplate code that researchers and practitioners need to write.
It also provides a number of advanced features, such as automatic precision scaling, distributed training, and multi-GPU
training, which can significantly speed up the training process and improve model accuracy.

PyTorch Lightning is a powerful tool for deep learning practitioners and researchers who want to focus on the high-level
aspects of their models, while benefiting from the best practices and optimizations provided by the PyTorch Lightning
framework.

### Hugging face

Hugging Face is an open-source software company that is dedicated to building state-of-the-art tools and libraries for
natural language processing (NLP). They provide a variety of tools for working with text data, including machine
learning models, datasets, and other NLP utilities.

### BERT models

BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained language model developed by Google in
2018 for natural language processing (NLP) tasks. It is based on the Transformer architecture, which is a neural network
architecture designed specifically for sequence-to-sequence tasks.

BERT is pre-trained on a large corpus of text data, and the pre-training process involves training the model on two
tasks: masked language modeling and next sentence prediction. In the masked language modeling task, the model is trained
to predict missing words in a sentence, while in the next sentence prediction task, the model is trained to predict
whether two sentences are consecutive or not.

After pre-training, the BERT model can be fine-tuned on a specific NLP task such as text classification,
question answering, or language generation. Fine-tuning involves training the last layer of the BERT model on
the task-specific data, while keeping the pre-trained weights fixed. This allows the model to learn task-specific
representations that can be used for downstream applications.

## Dataset

Here we rely mostly on standard methods. We used [Toxic comment dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) from kaggle toxic comment challenge which contains toxic comments.

## Code Structure

This example follows the PyTorch Lightning framework. For a short introduction to their
individual models see their
[15-minute pitch](https://lightning.ai/docs/pytorch/stable/starter/introduction.html).
Our code is structured into three different files.

- `data.py`: contains the data module, which is responsible for handling and loading the data to feed our module.
- `model.py` contains the model module, which defines the control flow of our training procedure.
- `main.py` contains the run and trainer configuration, which triggers the start of training, handles initialization of the respective objects and serves as a general entrypoint.

Isolating different tasks and responsibilities into single files promotes readability
and encapsulation. The defined classes and functions can easily be imported to different
modules.

### Model Module

Our `ToxicCommentTagger` inherits from The `LightningModule` class of PyTorch Lightning.
This serves to provide a standardized interface for defining machine learning models.
It needs to implement methods for e.g. the model's forward pass, loss function, and
optimization algorithm. It also provides hooks for various stages of the training
process, such as the training and validation steps, as well as methods for saving and
loading models.

In our case the `__init__` function initializes our neural network, which
is set to be `BERT` pretrained model by default. Furthermore, a linear layer is added as classification layer.

In our case we directly define `BCEWithLogitsLoss` as our loss function, which is the
correct choice for binary classification.

A topic that is also very important is tracking the training progress via metrics.
The `torchmetrics` package gives convenient wrappers for a majority of common
evaluation metrics. The metric objects handle collection of the metric in each
training step and subsequent aggregation across an epoch.
A major advantage of using `torchmetrics` is the direct integration into the Lightning
framework, which e.g. automatically reduces the collected metric elements after
finishing an epoch.

Our `training_step`, `validation_step`, `test_step` is set up rather trivial, as we
only do a simple model forward pass, compute the loss and collect our metrics.

The `configure_optimizers` function serves to initialize the optimizer of choice.
Often, the `AdamW` optimizer is used, which gives a solid and often quite optimal
baseline without setting specific hyperparameters. As it happens, the state-of-the-art
performance for standard vision dataset is usually achieved with the classic
stochastic gradient descent (`SGD`) paired with momentum and a learning rate scheduler.

### Data Module

The data module is used to organize and prepare data. It is responsible for loading,
preprocessing, and transforming the data in a desired format, usually to create batches
for neural network training.

Our `ToxicCommentDataModule` inherits from The `LightningDataModule` class of PyTorch Lightning.
The `LightningDataModule` provides a convenient interface for organizing and preparing
data for a PyTorch Lightning training or evaluation loop.
It allows users to separate the data loading and processing logic from the model itself,
making it easier to swap out different datasets or data processing pipelines.

In the `__init__` function we set a tokenizer that will be applied to every data
sample.

The `prepare_data` function is used for an initial setup like downloading the data
from a server and unpacking it. Here, we do exactly that.

The data container `ToxicCommentsDataset` inherits from the `datasets` API from torchvision.
In this function we convert the strings to sub-word strings. Here we used pretrained
[Fasttokenizer](https://huggingface.co/docs/transformers/main_classes/tokenizer) from Huggingface.
This function helps us with tokenization and encoder outputs and also how to deal with special
characters.

The `setup` function does the actual initialization of the dataset instances. We perform
a split of the dataset to provide an additional `validation` dataset apart from our
target `test` dataset.

Further, the `LightningDataModule` requires to implement methods, that return a
dataloader for each phase of the Lightning Trainer, which is: `train_dataloader`,
`val_dataloader` and `test_dataloader`.


### Main Module

The `main.py` usually serves as a general entrypoint for executing scripts or calling
modules. Using the `argparse` utility, we define a quick commandline interface, which
allows to change parameters without altering the actual code.

In our case, we include following options:

- `batch_size`: Number of data samples in each training step.
- `epochs`: Number of iterations over the dataset during training.
- `bert_model_name`: BERT pretrained model from hugging face.
- `data_dir`: Directory that contains the data or where to data should be downloaded to.
- `log_dir`: Directory where checkpoints and metrics shall be saved.
- `max_token_count`: Maximum number of tokens for tokenizer.
- `n_workers`: Number of concurrent workers for handling data fetching.

We rely on pretrained BERT as a language model that provides text embeddings for down stream task (here text classification). To get best results, you should unfreeze the weights in BERT model but if you have limited computation power you can freeze weights in pretrained model.
Call the main script to start training:

```shell
python berd/text_classification/main.py
```

This way, all default parameters are applied. If you e.g. want to train for more epochs
or use a different batch size, specify a flag:

```shell
python berd/text_classification/main.py --epochs 50 --batch_size 64
```

Additionally, all commandline options can be displayed by setting the `--help` flag
as it is shown in the example above.

Our `main` function first define the tokenizer from BERTtokenizer then it initializes the `ToxicCommentDataModule` and `ToxicCommentTagger`
module.

For each run, we define a directory that is named by the current timestamp, which
makes it unique and allows to compare different runs in future iterations.

Lastly, PyTorch Lightning defines the `Trainer` as a convenience class to provide
an easy training process, if one has specified a model and data module.
The `Trainer` has a myriad of options, so please refer the
[official documentation](https://lightning.ai/docs/pytorch/stable/common/trainer.html)
for an extended introduction.
A very important feature is the possibility to add callbacks that are mostly executed
after the end of an epoch. This gives additional flexibility and options to interact
with the otherwise pre-defined trainer object. We add `ModelCheckpoint` for saving
the best model according the accuracy on the validation set, `EarlyStopping` to stop
training if no improvement in the validation loss is happening and `TensorBoardLogger`
to save metrics persistently.
With the logger callback, the training can be tracked during its execution.
Just initialize the tensorboard service over

```shell
tensorboard --logdir <path/to/logdir>
```
and click on the presented link.

Eventually, the training is started over `.fit()` and our performance of the test
set is evaluated via `.test()`.



## Acknowledgements

- [Toxic comment dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- [Text classification](https://curiousily.com/posts/multi-label-text-classification-with-bert-and-pytorch-lightning/)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/cifar10-baseline.html)
- [Optimization](https://pytorch.org/docs/stable/optim.html)
- [Scheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
