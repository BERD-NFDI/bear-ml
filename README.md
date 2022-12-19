[![BERD License](https://img.shields.io/badge/license-BERD-yellowgreen)](https://www.berd-nfdi.de/)
![python](https://img.shields.io/badge/Python-3.10-brightgreen)

<p align="center">
<img src=assets/berd_logo.png  alt="berd logo"/>
</p>

# BERD Projects


This repository contains example use-cases and sub-projects to showcase the
functionality of deep learning training as well as related topics.

## 0. Contribution

New ideas and improvements are always welcome. Feel free to open an issue or contribute
over a pull request.
Our repository has a few automatic checks in place that ensure a compliance with PEP8 and static
typing.
It is recommended to use `pre-commit` as a utility to adhere to the GitHub actions hooks
beforehand.
First, install the package over pip and then set a hook:
```shell
pip install pre-commit
pre-commit install
```

## 1. Usage

First, clone the repo and change to the project directory.

```shell
git clone https://github.com/BERD-NFDI/BERD-projects.git
cd BERD-projects
```

The relevant use-cases and source codes are located in `berd`.
Currently, we support **python >= 3.10**.
It is recommended to install the required dependencies in a separate environment, e.g.
via `conda`.
A simpler alternative is a virtual environment, which is created and activated with:

```shell
python -m venv .venv
source .venv/bin/activate
```

Dependencies are then installed via `pip`.

```shell
pip install -r requirements.txt
```

The `berd` project is structured like a python package, which has the advantage of
being able to **install** it and thus reuse modules or functions without worrying about
absolute filepaths.
An editable version of `berd` is also installed over `pip`:

```shell
pip install -e .
```
