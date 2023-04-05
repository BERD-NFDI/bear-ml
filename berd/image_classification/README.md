# Image Classification with PyTorch Lightning

This repository is for the BERD project. It contains the example codes for Image Classification with PyTorch Lightning.



## Table of contents
* [Image Classification](#Image-Classification)
* [Deep learning](#Deep-learning)
* [Model](#Model)
* [Optimization](#Optimization)
* [Scheduler](#Scheduler)
* [PyTorch Lightning](#PyTorch-Lightning)
* [Dataset](#Dataset)
* [Usage/Examples](#Usage/Examples)
* [Acknowledgements](#Acknowledgements)
* [Feedback](#Feedback)


## Image Classification

Image classification is a computer vision task that involves categorizing an image into one of several predefined
classes or categories. This is typically done by training a machine learning model on a large dataset of images, where
each image is labeled with its corresponding class. The model then uses these examples to learn to recognize patterns
in the images that are associated with each class. Once the model has been trained, it can be used to predict the class
of new, unseen images based on the patterns it has learned. Image classification is widely used in a variety of
applications, including object recognition, face recognition, and medical imaging.


## Deep learning

Deep learning is a subset of machine learning that uses artificial neural networks to model and solve complex problems.
Deep learning models are designed to automatically learn hierarchical representations of data by processing it through
a series of non-linear transformations.

## Model

### Encoder

An encoder is a type of neural network that is used to extract important features from raw input data such as images,
videos or other visual data. The main purpose of an encoder is to convert the input data into a compressed representation
that captures the most important aspects of the input data.

An encoder typically consists of a series of convolutional layers, which apply filters to the input data to identify
patterns and features at different levels of abstraction. The output of each convolutional layer is then passed through
a non-linear activation function to introduce non-linearity into the model.

The final output of the encoder is a compressed representation of the input data, which can then be used as input to
a decoder network that reconstructs the original input data. Alternatively, the compressed representation can be used as
input to other downstream tasks such as classification, object detection or image segmentation.

### Resnet34

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


## Optimization

Optimization in deep learning refers to the process of the weights and biases of the neural network so that it can
accurately predict the target variable.

n order to optimize a deep learning model, an algorithm is used to iteratively update the model's parameters based on
the gradients of the loss function with respect to those parameters. This is typically done using stochastic gradient
descent (SGD) or one of its variants, such as Adam, Adagrad, or RMSProp.

The optimization process involves choosing appropriate hyperparameters, such as the learning rate and batch size, that
control the rate of parameter updates and the size of the mini-batches used during training. It also involves monitoring
the training and validation loss to prevent overfitting, which occurs when the model becomes too complex and starts to
memorize the training data instead of learning general patterns that can be applied to new data.

## Scheduler

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

## PyTorch Lightning

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


## Usage/Examples

In this example, we showcase how training a classifier works with pytorch lightning.

- `data.py` contains the data module.
- `model.py` contains the model module.
- `main.py` contains the run and trainer configuration
Call the main script to start training:

```shell
python berd/image_classification/main.py
```

There a few command line options available, which can be displayed by adding the
`--help` flag to the above command.

### Data module

the data module is a module that is used to organize and prepare data for a machine learning model. It is responsible
for loading, preprocessing, and transforming the data in a format that can be used by the model.

The data module in PyTorch Lightning provides a standardized interface for data loading and processing. It includes
a DataModule class that defines the data loading and processing pipeline, as well as methods for splitting the data
into training, validation, and test sets.

#### Prepare_data

This function serves for initial data preparation only. No assignments should be made here. In our case, we will just
download the data. In a multi GPU setting this is called with one single CPU process only.

#### Setup

This function is called by every process in a multi GPU setting and is dependent on the `stage` in which the lightning
trainer is currently in. Assignments should be made here. The function serves well e.g. for train/test/val splits or
initializing datasets.

#### Train/val/test dataloader

These functions return train/val/test dataloader.

### Model module

The LightningModule class in PyTorch Lightning provides a standardized interface for defining machine learning models.
It includes methods for defining the model's forward pass, loss function, and optimization algorithm. It also provides
hooks for various stages of the training process, such as the training and validation steps, as well as methods for
saving and loading models.

#### Forward

In this function the forward pass of the model is defined.

#### Training/validation/test_step

In these functions, training/validation/test step are defined.

#### Configure_optimizers

In this function, the optimizer is specified furthermore you can specify a learning-rate scheduler as well.

### Main

In this file, different configurations are specified, and the trainer and the run are defined to run the model.

## Acknowledgements

 - [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/cifar10-baseline.html)
 - [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
 - [Optimization](https://pytorch.org/docs/stable/optim.html)
 - [Scheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)


## Feedback

If you have any feedback, please reach out to us at ... .
