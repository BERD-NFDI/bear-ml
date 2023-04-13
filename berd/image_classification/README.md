# Image Classification with PyTorch Lightning

This repository is for the BERD project. It contains the example codes for Image Classification with PyTorch Lightning.

## Table of contents
* [Background](#background)
* [Dataset](#dataset)
* [Code Structure](#code-structure)
* [Acknowledgements](#Acknowledgements)

## Background

### Image Classification

Image classification is a computer vision task that involves categorizing an image into one of several predefined
classes or categories. This is typically done by training a machine learning model on a large dataset of images, where
each image is labeled with its corresponding class. The model then uses these examples to learn to recognize patterns
in the images that are associated with each class. Once the model has been trained, it can be used to predict the class
of new, unseen images based on the patterns it has learned. Image classification is widely used in a variety of
applications, including object recognition, face recognition, and medical imaging.


### Deep Learning

Deep learning is a subset of machine learning that uses artificial neural networks to model and solve complex problems.
Deep learning models are designed to automatically learn hierarchical representations of data by processing it through
a series of non-linear transformations.

### Neural Network Model

#### Encoder

An encoder is a type of neural network that is used to extract important features from raw input data such as images,
videos or other visual data. The main purpose of an encoder is to convert the input data into a compressed representation
that captures the most important aspects of the input data.

An encoder typically consists of a series of convolutional layers, which apply filters to the input data to identify
patterns and features at different levels of abstraction. The output of each convolutional layer is then passed through
a non-linear activation function to introduce non-linearity into the model.

The final output of the encoder is a compressed representation of the input data, which can then be used as input to
a decoder network that reconstructs the original input data. Alternatively, the compressed representation can be used as
input to other downstream tasks such as classification, object detection or image segmentation.

#### Resnet34

The ResNet34 architecture consists of 34 layers, including 33 convolutional layers and 1 fully connected layer. The first
layer of the network is a convolutional layer with 64 filters and a kernel size of 7x7, followed by a max-pooling layer
with a stride of 2. The following layers are arranged into a series of residual blocks, each containing multiple convolutional
layers with skip connections.

The skip connections in ResNet34 allow for easier training of deep networks by mitigating the vanishing gradient problem.
Specifically, the skip connections allow the gradient to flow through the network more easily, making it easier for the
model to learn from the data.

At the end of the residual blocks, the output is passed through a global average pooling layer, which averages the
spatial dimensions of the feature maps to produce a fixed-length feature vector. This feature vector is then passed
through a fully connected layer with a softmax activation function, which produces a probability distribution over
the different classes in the classification task.


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

## Dataset

The `CIFAR10` dataset from `torchvision` is used, and we utilize the `torch.hub` to download model definitions.

The CIFAR-10 dataset is a popular image classification dataset that consists of 60,000 32x32 color images in 10 classes,
with 6,000 images per class. The classes are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

It has been used extensively in the development and evaluation of deep learning models
and is often used as a baseline dataset for new research in the field.

If you are using the CIFAR-10 dataset in your research or application, it is important
to note that the images are relatively low resolution and may require preprocessing
before use. Additionally, due to its popularity, there are many existing pre-trained
models and benchmarks available for comparison.

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

Our `ClassifierModule` inherits from The `LightningModule` class of PyTorch Lightning.
This serves to provide a standardized interface for defining machine learning models.
It needs to implement methods for e.g. the model's forward pass, loss function, and
optimization algorithm. It also provides hooks for various stages of the training
process, such as the training and validation steps, as well as methods for saving and
loading models.

In our case the `__init__` function initializes our neural network via a string, which
is set to be `resnet34` by default. Using the PyTorch hub, the full model structure
is immediately downloaded and initialized. Optionally, we could initialize the model
with weights from classifying imagenet.

In our case we directly define `CrossEntropyLoss` as our loss function, which is the
correct choice for multi-class classification.

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
Often, the `Adam` optimizer is used, which gives a solid and often quite optimal
baseline without setting specific hyperparameters. As it happens, the state-of-the-art
performance for standard vision dataset is usually achieved with the classic
stochastic gradient descent (`SGD`) paired with momentum and a learning rate scheduler.

### Data Module

The data module is used to organize and prepare data. It is responsible for loading,
preprocessing, and transforming the data in a desired format, usually to create batches
for neural network training.

Our `CIFAR10DataModule` inherits from The `LightningDataModule` class of PyTorch Lightning.
The `LightningDataModule` provides a convenient interface for organizing and preparing
data for a PyTorch Lightning training or evaluation loop.
It allows users to separate the data loading and processing logic from the model itself,
making it easier to swap out different datasets or data processing pipelines.

In our constructor we set a base transformation that will be applied to every data
sample. Here, we will convert the 8-bit CIFAR10 PIL image to a normalized tensor.
Additionally, we apply scaling to the image, which results in pixel values having a
mean of zero and a standard deviation of one. This has been shown to stabilize the
training process and helps in converging the model.
In the `__init__` function we offer to add some more data transformations, which can
be used for setting data augmentations.

The `prepare_data` function is used for an initial setup like downloading the data
from a server and unpacking it. Here, we do exactly that.

The actual data container `CIFAR10` is provided by the `datasets` API from torchvision.
When using your own datasets, one usually needs to define their own container objects.
As a matter of convenience, we do that in the `CIFAR10Wrapper` class, which does not
contain much logic, but allows to flexibly set data transformation after the
initialization of a `CIFAR10` instance.

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
- `model_id`: Model identifier for a model in the PyTorch hub.
- `data_dir`: Directory that contains the data or where to data should be downloaded to.
- `log_dir`: Directory where checkpoints and metrics shall be saved.
- `lr`: Learning rate for the optimizer.
- `wd`: L2 regularization.
- `n_cls`: Number of classes in the training dataset.
- `n_workers`: Number of concurrent workers for handling data fetching.

The script itself can be executed via:

```shell
python berd/image_classification/main.py
```

This way, all default parameters are applied. If you e.g. want to train for more epochs
or use a different model, specify a flag:

```shell
python berd/image_classification/main.py --epochs 50 --model_id resnet 50
```

Additionally, all commandline options can be displayed by setting the `--help` flag
as it is shown in the example above.

Our `main` function first initializes the `ClassifierModule` and `CIFAR10DataModule`
module. Notably, we add a data augmentation policy provided by PyTorch in the
constructor, which hopefully improves our performance.

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

 - [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/cifar10-baseline.html)
 - [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
 - [Optimization](https://pytorch.org/docs/stable/optim.html)
 - [Scheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
