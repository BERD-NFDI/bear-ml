# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tutorial for Splitting Datasets - Part 2
#
# The first part of this tutorial concerned with the general concept of splitting
# tabular data contained in pandas dataframes. In the second part, we would like
# to focus on the `Dataset` structure, which is provided by PyTorch as a
# measure for providing data to the neural network training procedure.
#
# |                  |                                                                                                                                                                                                                                                                  |
# |------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
# | Requirements   	 | 	- Basic python skills                                                                                                                                                                                                                                           |
# | Learning Goals 	 | 	- Understanding basics of a pandas dataframe <br/>- Concept of splitting data into multiple partitions <br/>- Various splitting strategies for dataframes. <br/>- Understanding cross-validation <br/>- Application of cross-validation in a practical use-case |
# | Limitations    	 | - The tutorial only handles pandas dataframes and a numpy array in the practical use-case.                                                                                                                                                                       |
#

import uuid

# %% is_executing=true
from typing import Any, List

import torch
from torch.utils.data import Dataset

# %% [markdown]
# ## The Dataset Object
#
# Before we dive into the actual data splitting, we introduce the `Dataset` object in general.
# While there exists the possibility to directly create a dataset from an array (`TensorDataset`)
# or a directory filled with images (`ImageFolder` in `torchvision`), building a custom routine
# yields maximal flexibility and requires only a little setup.
#
# As shown in the code snippet below, creating a dataset requires the implementation of two
# functions. The most important one is `__getitem__`. The function will receive an `idx` starting
# from 0 and should return the training data corresponding to the index. The kind of return value
# is not fixed or limited in any case. Returned tensors would be batched automatically by the
# subsequent `Dataloader`, while more complex data types would require a custom collate function.
# However, this is an advanced topic and is not covered in this tutorial.
#
# The second function to implement is the `__len__` function, which should simply return the
# number of elements in the dataset, which is very helpful for the dataloader.


# %%
class MyDataset(Dataset):
    def __len__(self) -> int:
        pass

    def __getitem__(self, idx: int) -> Any:
        pass


# %% [markdown]
# To showcase the base functionality of a `Dataset`, we create an object that
# contains a list of 100 randomly generated strings as our dummy data.
# The `__len__` function simply returns the number of elements in our data list.
# The `__getitem__` function is tied to the indices of the list.


# %%
class MyDataset(Dataset):
    def __init__(self) -> None:
        num_samples = 100
        # This just generates a list of random strings
        self.data = [uuid.uuid4().hex.upper()[0:6] for _ in range(num_samples)]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> str:
        return self.data[idx]


# %% [markdown]
# `__len__` and `__getitem__` belong to the group of *magic methods*. This implies
# that the keyword name of the function is reserved to work with another operator.
# In other words, when calling the `len` function on our dataset object, the
# `__len__` function is called internally.
# The `__getitem__` method is triggered by the bracket operator as it is usual in
# dictionaries or list accesses.

# %%
ds = MyDataset()
print(f'Number of samples in dataset: {len(ds)}')
print('Samples:', ds[0], ds[42], ds[77])

# %% [markdown]
# As shown above we can extract samples of our choice with the help of the `[]` operator.
#
# In the next steps we show some possibilities of splitting and organizing such a `Dataset`
# object. We make the differentiation between *internal* and *external* splitting.
#
# ## Internal Splitting
#
# With interal splitting we refer to the idea of subsetting the dataset itself.
# An easy-to-use utility is hereby the `random_split` function, which is contained in PyTorch
# itself.
# This functions returns two instances of `Dataset`.
#

# %%
from torch.utils.data import random_split

train_ds, test_ds = random_split(dataset=ds, lengths=(0.8, 0.2))

print(f'Samples in train dataset: {len(train_ds)}')
print(f'Samples in test dataset: {len(test_ds)}')

# %% [markdown]
# However, without setting a seed, the split is not reproducible.
# Everytime the function is called, the subsets contain different samples.

# %%
for i in range(5):
    train_ds, test_ds = random_split(dataset=ds, lengths=(0.8, 0.2))
    print(f'({i}): {train_ds[0]} {train_ds[42]} {train_ds[77]}')

# %% [markdown]
# In PyTorch the preferred way of setting seeds is by using the `Generator` utility.
# This object keeps track of the random state and can be used to steer random behavior
# without setting a global context.
#
# For this, a `Generator` instance is created and we set our own seed with the
# `manual_seed` method.
# It is quite often that seeds are chosen to be some prominent numbers like `1337` or `42`.
# In contrast to this common practice, the documentation of the PyTorch generator recommends
# the following:
#
# *It is recommended to set a large seed, i.e. a number that has a good balance of 0 and 1 bits. Avoid having many 0 bits in the seed.*
#
# As shown in literature before, the choice of the seed can have a significant impact on your result.
# So it may be a good opportunity to use a large number.

# %%
from torch import Generator

for i in range(5):
    gen = Generator().manual_seed(1337)
    train_ds, test_ds = random_split(ds, (0.8, 0.2), generator=gen)
    print(f'({i}): {train_ds[0]} {train_ds[42]} {train_ds[77]}')

# %% [markdown]
# Note that each time when the split is done, we create a new instance of `Generator` with the same seed.
# If `gen` is called, its internal state progresses and invoking the split operation is based on different
# random states.
#
# As it may sound counterintuitive this is an expected behavior.
# In this way randomness within the loop is still given, but e.g. doing 5 different splits in a row itself
# is reproducible.

# %%
gen = Generator().manual_seed(1337)

for i in range(5):
    train_ds, test_ds = random_split(ds, (0.8, 0.2), generator=gen)
    print(f'({i}): {train_ds[0]} {train_ds[42]} {train_ds[77]}')

# %% [markdown]
# If one only wants to instantiate the `Generator` object once and would like to have a reproducible
# random behaviour within the program, the current random state can be extracted using `get_state`.
# The saved state can then be established again by `set_state`.

# %%
gen = Generator().manual_seed(1337)
gen_state = gen.get_state()

for i in range(5):
    train_ds, test_ds = random_split(ds, (0.8, 0.2), generator=gen)
    gen.set_state(gen_state)
    print(f'({i}): {train_ds[0]} {train_ds[42]} {train_ds[77]}')

# %% [markdown]
# In our examples we only did a train and validation split.
# The random split function additionally does arbitrary splits, when a respective `lengths` sequence is provided.
# In our case, we can easily use this to do the previously discussed train, validation and test split.

# %%
train_ds, val_ds, test_ds = random_split(dataset=ds, lengths=(0.6, 0.2, 0.2))

# %% [markdown]
# To check which samples are in the respective subset, the `indices` attribute helps to identify the index of the samples in the original `Dataset` object.
# These indices could be saved persistently, e.g. in a file, which in turn can be used to reconstruct the training and test splits or - in other words -
# make your experiments reproducible.

# %%
train_idxs = train_ds.indices
print(train_idxs)

# %% [markdown]
# Given a list of indices and the original dataset, the subset can be reconstructed using the `Subset` object:

# %%
from torch.utils.data import Subset

train_ds = Subset(dataset=ds, indices=train_idxs)


# %% [markdown]
# While subsetting the full `Dataset` object is a possibility, it also has its limitations.
# Keeping in mind the first part of this series, there are more options to
# splitting and subsetting than pure randomization.
# If one would like to utilize other utilities like stratified splitting or cross-validation,
# a different method needs to be used.
#
# ## External Splitting
#
# Alternatively, the dataset itself can be constructed by passing data from the outside.
# A trivial thing to do would be to split the data and keep a copy of every split.
# However, in most cases keeping multiple copies of your data is inefficient or
# infeasible (thinking of most computer vision datasets).
#
# A solution to this problem is by handling data access over keys.
# A key can be anything that uniquely identifies a data sample: An index in a pandas dataframe,
# a file path to an image, a patient id, etc.
# Using keys, the dataset object serves as a measure to provide and/or load data on demand
# without having multiple copies of it.
# This lightweight list of keys can also be utilized for cross-validation to determine
# different folds.


# %%
class MyDataset(Dataset):
    def __init__(self, keys: List[str], *args, **kwargs) -> None:
        # Set list of keys, that identify a datasample.
        self.keys = keys

    # Function that loads/provides a datasample given a key.
    def load_item(self, key) -> Any:
        pass

    # The size of the dataset is the number of keys.
    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> Any:
        # Given an index, return a keys.
        key = self.keys[idx]
        # Retrieve data sample from key.
        return self.load_item(key)
