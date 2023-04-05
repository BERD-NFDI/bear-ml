# Image Segmentation with PyTorch Lightning

This repository is for the BERD project. It contains the example codes for Image Segmentation.



## Table of contents
* [Image Segmentation](#Image-Segmentation)
* [Deep learning](#Deep-learning)
* [Model](#Model)
* [Optimization](#Optimization)
* [Scheduler](#Scheduler)
* [Segmentation Models PyTorch](#Segmentation-Models-PyTorch)
* [Dataset](#Dataset)
* [Usage/Examples](#Usage/Examples)
* [Acknowledgements](#Acknowledgements)
* [Feedback](#Feedback)


## Image Segmentation

Image segmentation is the process of dividing an image into multiple segments or regions, where each segment represents
a distinct object or part of the image. The goal of image segmentation is to simplify or change the representation of
an image into something that is more meaningful and easier to analyze.

The segmentation can be performed based on various features of an image, such as color, texture, shape, or intensity.
The output of an image segmentation algorithm is typically a binary mask that labels each pixel in the image as belonging
to a particular segment or background.


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

### Decoder

a decoder is a neural network module that takes a high-level representation of an image learned by the encoder and
produces a lower-resolution version of the original input image.

The decoder is responsible for producing an output image from the encoded representation, by upsampling or otherwise
expanding the compressed representation back to the original input image size. The decoder can use various techniques
to perform this upsampling, such as transpose convolution, nearest-neighbor interpolation, or bilinear interpolation.

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

## Segmentation Models PyTorch

Segmentation Models PyTorch is a Python package that provides implementations of various image segmentation models for
the PyTorch deep learning framework. The package includes popular segmentation architectures such as UNet, LinkNet,
PSPNet, and FPN, as well as their variants.

The Segmentation Models PyTorch package provides a simple and easy-to-use interface to train and evaluate image
segmentation models. The package includes pre-trained models and allows fine-tuning of these models on custom datasets.
The models can be trained using various loss functions, such as binary cross-entropy, dice loss, or focal loss.

The package also provides utilities for data preprocessing, visualization of segmentation masks, and prediction of
segmentation masks for new images. Segmentation Models PyTorch is designed to work seamlessly with other PyTorch packages
and libraries, such as torchvision and torchsummary.

## Dataset

we use the Oxford pet dataset, which has around 200 images for each of its 37 cats & dogs species.

## Usage/Examples

### Details how to run/modify the code for image segmentation

The backbone of this showcase is based on the `Segmentation Models PyTorch` package,
which provides a library with standard implementations and API of common segmentation
algorithms.


### Modify  the pretrained model architecture.

You can choose other pretrained models and use them.
This is done over changing the `arch` and `encoder` params in the `SegmentationModel`
class.
Available segmentation architectures are listed [here](https://smp.readthedocs.io/en/latest/models.html) and
encoders are [here](https://smp.readthedocs.io/en/latest/encoders.html).

### Commandline arguments

Different hyperparameters can lead to better results. Among others, you can modify
a few of them as flags when calling the script, e.g.:

- `--lr` : Learning-rate
- `--batch_size`: Batch-size
- `--wd`: Weight decay

Moreover, by setting:

- `--freeze_encoder`
- `--freeze_head`
- `--freeze_decoder`

certain parts of the U-net shaped models can be frozen, i.e. prohibited from gradient
updates. This plays a major role in preventing overfitting and finetuning with fewer
resources. Try a few combinations and see how it affects your training! Often it is
enough to only train the `head` or the `head` plus `decoder`.


## Acknowledgements

 - [Segmentation Models PyTorch](https://segmentation-modelspytorch.readthedocs.io/en/latest/#)
 - [The Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/)
 - [Optimization](https://pytorch.org/docs/stable/optim.html)
 - [Scheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)


## Feedback

If you have any feedback, please reach out to us at ... .
