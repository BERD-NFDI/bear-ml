# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# flake8: noqa
# noqa

# %% [markdown]
# # Tutorial for Configuration of Experiments
#
# This tutorial aims to provide a variety of insights on how to parametrize
# experiments in python scripts.
#

# %%
import math

# %% [markdown]
# As a starter let's assume our machine learning project consists solely on
# computing the result of a sine function.


# %%
def compute_function(x):
    return 1.5 * math.sin(2 * math.pi / 0.6 * (x - 3)) + 0.2


# %% [markdown]
# ## Refactoring the base function
#
# This function seems pretty easy right? But aside from adding proper documentation,
# how can we make the code more self-explanatory?
# Suddenly appearing literals / constants are called **magic numbers** and
# should be avoided.
# As a first step, we assign variables as our magic numbers.


# %%
def compute_function(x):
    a = 1.5
    b = 0.6
    c = 3
    d = 0.2
    return a * math.sin(2 * math.pi / b * (x - c)) + d


# %% [markdown]
# This does not really help! Variable names are important, and they should be unique
# , and it should be obvious what they represent.
# So let's try again:


# %%
def compute_function(x):
    amplitude = 1.5
    period = 0.6
    shift = 3
    center = 0.2
    return amplitude * math.sin(2 * math.pi / period * (x - shift)) + center


# %% [markdown]
# It is more verbose now, but it is way more intuitive than before.
# But what happens if you want to change the values for your experiments?
# We could edit the code, but this alters the function itself,
# which is also a bad practice.
# An easy fix is to allow function parameters and set the above values as defaults.


# %%
def compute_function(
    x,
    amplitude=1.5,
    period=0.6,
    shift=3,
    center=0.2,
):
    return amplitude * math.sin(2 * math.pi / period * (x - shift)) + center


# %% [markdown]
# This gives way more flexibility without altering the actual code of the function.
# Additionally, it is considered a good practice if type hints are added.
# These indicate the datatype of the parameters as well as the function return value
# and contribute a lot to readability, maintainability and error prevention.


# %%
def compute_function(
    x: float,
    amplitude: float = 1.5,
    period: float = 0.6,
    shift: float = 3,
    center: float = 0.2,
) -> float:
    return amplitude * math.sin(2 * math.pi / period * (x - shift)) + center


# %% [markdown]
# ## Declaring constants
#
# Imagine you use the function somewhere in a script e.g. like:


# %%
def main() -> None:
    # <--- CODE ABOVE --->
    x_val = 0.75
    y = compute_function(x=x_val, shift=-2.5)
    # <--- CODE BELOW --->


# %% [markdown]
# We run into the problem of magic numbers again!
# Suddenly, a `-2.5` appears and if one doesn't read the full script carefully,
# we miss that our default function behavior was altered.
# If we want to explicitly inform the reader that we changed some parameters once,
# defining **constants** is a good idea.
# Constants are usually placed in the beginning of the script after
# the import statements and are written in capslock.
# Sadly, python doesn't allow any real immutable constants,
# but at least we can add a type hint indicating that we won't change
# the value of our constant again.

# %%
from typing import Final

SINUS_SHIFT: Final[float] = -2.5


def main() -> None:
    # <--- CODE ABOVE --->
    x_val = 0.75
    y = compute_function(x=x_val, shift=SINUS_SHIFT)
    # <--- CODE BELOW --->


# %% [markdown]
# With this mechanic in place, we have added a quick way of changing our parameters
# manually in one place without having to scroll through the full script.
# We can also import constants into other modules.
#
# An alternative to defining constants is the python utility of `dataclasses`.
# Dataclasses are convenience classes to store attributes and
# are initialized over the decorator `@dataclass`.
# By setting `frozen=True` an exception is raised, when an assignment
# to a field was made, i.e. if we try to change an attribute of the data class.
# Additionally, we remove the `__init__` function,
# which disables changing of attributes over the constructor.
# Thus, we have some way to force constants and the class serves as a
# good container for all parameters that may be subject to change.

# %%
from dataclasses import dataclass


@dataclass(init=False, frozen=True)
class FunctionParameters:
    sinus_amplitude: float = 1.5
    sinus_period: float = 0.6
    sinus_shift: float = -2.5
    sinus_center: float = 0.2


# %% [markdown]
# A big advantage of the dataclass in comparison to a classic dictionary (without
# excessive type hinting) is the inclusion into most IDEs' autocomplete mechanic.
#
# We decide to pass our parameters as function argument to our main method:


# %%
def main(func_params: FunctionParameters) -> None:
    # <--- CODE ABOVE --->
    x_val = 0.75
    y = compute_function(
        x=x_val,
        amplitude=func_params.sinus_amplitude,
        period=func_params.sinus_period,
        shift=func_params.sinus_shift,
        center=func_params.sinus_center,
    )
    # <--- CODE BELOW --->


if __name__ == '__main__':
    func_params = FunctionParameters()
    main(func_params)

# %% [markdown]
# ## Parsing arguments
#
# A dataclass makes it more obvious and clear, what values are carried by some
# parameters, but the problem of having to edit source files if we change our
# setup remains and should be avoided altogether.
#
# A quick CLI interface is generated over the standard python `argparse` utility.
# We could just feed the obtained parameters as arguments to the main function,
# but to be clean from the start, we use them to initialize our dataclass.
# For this we need to remove the `init=False` argument in `FunctionParameters`.
# Additionally, we remove the default values from `FunctionParameters`,
# which has two effects:
# First, passing a parameter is now required when creating an
# instance of `FunctionParameters`.
# Second, the default arguments become our single source of truth.

# %%
import argparse


@dataclass(frozen=True)
class FunctionParameters:
    sinus_amplitude: float
    sinus_period: float
    sinus_shift: float
    sinus_center: float


def parse_args() -> FunctionParameters:
    parser = argparse.ArgumentParser()
    parser.add_argument('--amplitude', '-a', type=float, default=1.5)
    parser.add_argument('--period', '-p', type=float, default=0.6)
    parser.add_argument('--shift', '-s', type=float, default=3)
    parser.add_argument('--center', '-c', type=float, default=0.2)

    args = parser.parse_args()
    return FunctionParameters(
        sinus_amplitude=args.amplitude,
        sinus_period=args.period,
        sinus_shift=args.shift,
        sinus_center=args.center,
    )


def main(func_params: FunctionParameters) -> None:
    # <--- CODE ABOVE --->
    x_val = 0.75
    y = compute_function(
        x=x_val,
        amplitude=func_params.sinus_amplitude,
        period=func_params.sinus_period,
        shift=func_params.sinus_shift,
        center=func_params.sinus_center,
    )
    # <--- CODE BELOW --->


if __name__ == '__main__':
    func_params = parse_args()
    main(func_params)

# %% [markdown]
# By defining arguments as above, one could ca
