# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tutorial for Configuration of Experiments
#
# This tutorial aims to provide a variety of insights on how to parametrize
# experiments in python scripts.
#

# %% pycharm={"is_executing": true}
import math

# %% [markdown]
# As a starter let's assume our machine learning project consists solely on
# computing the result of a sine function.


# %% pycharm={"is_executing": true}
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


# %% pycharm={"is_executing": true}
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


# %% pycharm={"is_executing": true}
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


# %% pycharm={"is_executing": true}
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


# %% pycharm={"is_executing": true}
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


# %% pycharm={"is_executing": true}
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

# %% pycharm={"is_executing": true}
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

# %% pycharm={"is_executing": true}
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


# %% pycharm={"is_executing": true}
def main(func_params: FunctionParameters) -> None:
    x_val = 0.75
    y = compute_function(
        x=x_val,
        amplitude=func_params.sinus_amplitude,
        period=func_params.sinus_period,
        shift=func_params.sinus_shift,
        center=func_params.sinus_center,
    )


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

# %% pycharm={"is_executing": true}
# This simulates empty arguments.
# Otherwise, calling below code snippet would fail in a jupyter notebook.
import sys

sys.argv = ['']

# %% pycharm={"is_executing": true}
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
    x_val = 0.75
    y = compute_function(
        x=x_val,
        amplitude=func_params.sinus_amplitude,
        period=func_params.sinus_period,
        shift=func_params.sinus_shift,
        center=func_params.sinus_center,
    )


if __name__ == '__main__':
    func_params = parse_args()
    main(func_params)

# %% [markdown]
# `argparse` is included in the python standard library and is usually sufficient for most CLI tasks.
# Alternatively, one could utilize some convenience libraries as
# (click)[https://click.palletsprojects.com] or (fire)[https://google.github.io/python-fire/guide/].
# These usually convert function arguments directly or use the decorator functionality
# to declare CLI arguments.

# %% [markdown]
# ## Configuration Files
#
# Another way of passing arguments into a script is by using configuration files.
# Having external files to do the parametrization is often even more convenient and readable.
# Information is also stored persistently.
# Further, when doing sweeps, having configuration files allows for a distinct experiment setup,
# e.g. when scheduling runs over a directory of configuration files.
#
# Config files can come in various formats. Each format has its pros and cons but in the majority
# of basic use-cases it comes down to personal preference.
#
# In the following, we show examples of `JSON`, `YAML` and `TOML` paired with a way to parse the
# contents into a python dictionary.
#
#
# ### JSON
#
# `.json` files are a widely adapted standard for information exchange.
# While JSON or `Javascript Object Notation` is an established format providing
# a variety of options, the syntax is a bit more complicated and sensitive in
# comparison to other methods.
#
# Loading a `.json` file is done over the standard python `json` utility.
#
# ```json
# {
# 	"sinus": {
# 		"amplitude": 1.6,
# 		"period": 0.6,
# 		"shift": 3,
# 		"center": 0.2
# 	}
# }
# ```

# %% pycharm={"is_executing": true}
import json
from typing import Dict


def load_json(f_path: str) -> Dict:
    with open(f_path, 'r') as f:
        return json.load(f)


# %% [markdown]
# ### YAML
#
# YAML or `YAML Ain't Markup Language` follows the same key-value principles as JSON but
# has an easier syntax and is a popular method for configuration files.
#
# The `.yaml` file is loaded using the `pyyaml` package, which can be installed
# over `pip`.
#
# ```yaml
# sinus:
#     amplitude: 1.6
#     period: 0.6
#     shift: 3
#     center: 0.2
# ```

# %% pycharm={"is_executing": true}
import yaml


def load_yml(f_path: str) -> Dict:
    with open(f_path, 'r') as ymlfile:
        return dict(yaml.load(ymlfile, Loader=yaml.FullLoader))


# %% [markdown]
# ### TOML
#
# TOML or `Tom's Obvious Minimal Language` is designed to be a minimal
# configuration file format and is becomingly more adapted by various
# packages plus in the build process of python packages.
#
# There are various packages available to read TOML files.
# We simply choose the `toml` package.
#
# ```toml
# [sinus]
# amplitude = 1.6
# period = 0.6
# shift = 3
# center = 0.2
# ```

# %% pycharm={"is_executing": true}
import toml


def load_toml(f_path: str) -> Dict:
    return toml.load(f_path)


# %% [markdown]
# In the next step, we show how we can use one of the above function or formats to parse arguments
# in our python script.
# Again, we could simply parse the content of our files into dictionaries,
# but having a dedicated class for our parameters gives the benefit of e.g.
# type security, linting, prevention of key errors etc.
#
# As we saw in the file examples, the formats allow for hierarchical information,
# i.e. `sinus` as category.
# At a certain point it would probably make sense to structure dataclasses according
# to the respective hierarchy, but we save this step for another tutorial ;)
#
# As we see, we are still using argument parsing!
# This step is barely removable, because at one point we have to tell our program,
# where the configuration file is located at.
# Obviously, we can just hardcode it, but how do we set a config file at a
# different location?
#
# We stashed a matching config file in the repository beforehand, so we just need to emulate
# CLI arguments.

# %% pycharm={"is_executing": true}
sys.argv = ['', '--config', 'utils/configuration_tutorial.toml']

# %% pycharm={"is_executing": true}
import argparse


@dataclass(frozen=True)
class FunctionParameters:
    sinus_amplitude: float
    sinus_period: float
    sinus_shift: float
    sinus_center: float


def get_configuration() -> FunctionParameters:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='config.toml')
    args = parser.parse_args()

    cfg_dict = load_toml(args.config)
    return FunctionParameters(
        sinus_amplitude=cfg_dict['sinus']['amplitude'],
        sinus_period=cfg_dict['sinus']['period'],
        sinus_shift=cfg_dict['sinus']['shift'],
        sinus_center=cfg_dict['sinus']['center'],
    )


def main(func_params: FunctionParameters) -> None:
    x_val = 0.75
    y = compute_function(
        x=x_val,
        amplitude=func_params.sinus_amplitude,
        period=func_params.sinus_period,
        shift=func_params.sinus_shift,
        center=func_params.sinus_center,
    )


if __name__ == '__main__':
    func_params = get_configuration()
    main(func_params)

# %% [markdown]
# ## (Advanced) Automatic Parsing & Configuration File Checker
#
# Previously, we have set up a routine, which is able to read a config file
# to memory and use the content to initialize a type safe configuration object.
# However, with an increasing number of parameters, the whole process gets
# more and more extensive.
#
# In the following, we develop a function that extracts the attributes from a
# generic dataclass and ensures that all of them exist in the loaded
# configuration file.
#
# First, the `dataclasses` utility has a function that lists the attributes or
# `fields` in this context. We don't need to initialize the class for this.

# %% pycharm={"is_executing": true}
import dataclasses

cls_attributes = [field.name for field in dataclasses.fields(FunctionParameters)]
print(cls_attributes)

# %% [markdown]
# We load the config file to memory via the loading function defined above.

# %% pycharm={"is_executing": true}
f_path = 'utils/configuration_tutorial.toml'
cfg_dict = load_toml(f_path)

print(cfg_dict)

# %% [markdown]
# As we can see, we need to map the nested dictionary to the fieldnames.
# A simple solution is to employ a unique naming scheme and combining multiple
# levels via concatenation.
# This could get complicated very fast, if we allow unlimited hierarchies, which
# in turn would require a recursive parsing function.
# In our example, we set a depth limit of one.

# %% pycharm={"is_executing": true}
cfg_dict_new = {}
for key, val in cfg_dict.items():
    if isinstance(val, dict):
        for sub_key, sub_val in val.items():
            cfg_dict_new.update({key + '_' + sub_key: sub_val})
    else:
        cfg_dict_new.update({key: val})

print(cfg_dict_new)

# %% [markdown]
# Next, we check whether all parsed arguments match our required fields.
# We convert both objects into sets and compute difference. If all field names are contained,
# the result should have a length of zero.

# %% pycharm={"is_executing": true}
print(set(cls_attributes) - set(cfg_dict_new.keys()))

print(set(cls_attributes + ['xyz']) - set(cfg_dict_new.keys()))

# %% [markdown]
# To make sure that we only parse arguments, which can be processed by the constructor of the
# dataclass, we take the intersection of sets afterwards.

# %% pycharm={"is_executing": true}
final_keys = set(cls_attributes + ['xyz']).intersection(cfg_dict_new.keys())
print(final_keys)

# %% [markdown]
# Eventually, the contents of the filtered config file can be passed to the dataclass via the `**` operator.

# %% pycharm={"is_executing": true}
cfg_dict_filtered = {key: cfg_dict_new[key] for key in final_keys}
FunctionParameters(**cfg_dict_filtered)

# %% [markdown]
# Let's combine everything in one handy object!

# %% pycharm={"is_executing": true}
from dataclasses import is_dataclass

import toml


class ConfigurationParser:
    def __init__(self, cfg_cls=FunctionParameters) -> None:
        if not is_dataclass(cfg_cls):
            raise ValueError('{} is not a valid dataclass.'.format(cfg_cls))
        self.cfg_cls = cfg_cls

    def get_configuration(self, f_path: str):
        # Load TOML file
        cfg_dict = ConfigurationParser.load_toml(f_path)
        # Convert nested keys
        cfg_dict = ConfigurationParser.convert_config_hierarchy(cfg_dict)
        # Get dataclass attributes
        cls_attributes = [field.name for field in dataclasses.fields(self.cfg_cls)]

        # Check if all necessary keys are contained
        arg_diff = set(cls_attributes) - set(cfg_dict.keys())
        if len(arg_diff) > 0:
            raise ValueError(
                'Config file keys do not match all keys required by the dataclass. '
                'Missing keys: {}'.format(arg_diff)
            )

        # Initialize configuration object with filtered keys
        cfg_keys = set(cls_attributes).intersection(cfg_dict_new.keys())
        cfg_dict = {k: cfg_dict_new[k] for k in cfg_keys}
        return self.cfg_cls(**cfg_dict)

    @staticmethod
    def load_toml(f_path: str) -> Dict:
        return toml.load(f_path)

    @staticmethod
    def convert_config_hierarchy(cfg_dict: Dict) -> Dict:
        cfg_dict_new = {}
        for key, val in cfg_dict.items():
            if isinstance(val, dict):
                for sub_key, sub_val in val.items():
                    cfg_dict_new.update({key + '_' + sub_key: sub_val})
            else:
                cfg_dict_new.update({key: val})

        return cfg_dict_new


# %% pycharm={"is_executing": true}
f_path = 'utils/configuration_tutorial.toml'
ConfigurationParser().get_configuration(f_path)

# %% [markdown]
# Given our `ConfigurationParser` and `FunctionParameters` classes, we can finalize the configuration setup of our script:

# %% pycharm={"is_executing": true}
import argparse


def get_config_path() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='config.toml')
    args = parser.parse_args()
    return args.config


def main(func_params: FunctionParameters) -> None:
    x_val = 0.75
    y = compute_function(
        x=x_val,
        amplitude=func_params.sinus_amplitude,
        period=func_params.sinus_period,
        shift=func_params.sinus_shift,
        center=func_params.sinus_center,
    )


if __name__ == '__main__':
    cfg_path = get_config_path()
    func_params = ConfigurationParser(FunctionParameters).get_configuration(cfg_path)
    main(func_params)

# %% [markdown]
# Obviously, this kind of setup is a complete overkill for a small script,
# but let us imagine a larger project with 30+ parameters and multiple collaborates.
# Giving a secure interface with static typing is an important step towards reliability, readability and encapsulation.
