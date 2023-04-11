[![BERD License](https://img.shields.io/badge/license-BERD-yellowgreen)](https://www.berd-nfdi.de/)
![python](https://img.shields.io/badge/Python-3.10-brightgreen)

<p align="center">
<img src=assets/berd_logo.png  alt="berd logo"/>
</p>

# 0. Introduction & Scope

Introducing "BERD Projects": the go-to repository for novice users looking to learn
practical computing and data skills in the context of business and economics research.

As more and more business and economics researchers rely on digital tools and methods,
the need for accessible and effective training in these areas becomes increasingly
pressing. "BERD Projects" seeks to fill this gap by providing a repository of training
materials designed specifically for novice users who want to learn the fundamentals of
data management, analysis, and visualization in the context of business and
economics research.

Whether you are a graduate student, an early career researcher, or an experienced
professional looking to update your skills, "BERD Projects" can help you improve your
computing and data skills in a practical and effective way.

## Target Group

Applied researchers from Business, Economics, Social Sciences or related field, who
- .. have little or no formal eduction or experience with analysis pipelines for unstructured data.
- .. want to know how to appropriately structure the codebase for their data analysis project.
- .. seek advice for specific problems faced when analyzing unstructured data and training models (see our [Discussion board](./discussions)).

## Practical Lessons

The repository contains introductory units for various topics.

Currently available:

- [Parametrizing python scripts with config files](berd/basics/configuration_tutorial_nb.ipynb)
- [Linear regression with PyTorch](berd/basics/linear_regression_nb.ipynb)

If you feel that some important lesson is missing, please don't hesitate to contact us and we will happily try to add it.

## Educational Resources

We maintain a carefully curated list of educational resources intended to help researchers read up on specific topics.
This list includes, but is not limited to, (a) courses, (b) books, (c) blog posts, (d) great overview papers, and (e) talks/presentations.  
It is available under [this link](https://docs.google.com/document/d/1EH3Yq8Oi5wRq96t8IRjyMSKqj2WmihMclZiTZr7BmhA/edit?usp=sharing)

## Discussion Board

Thie repository is accompanied by a discussion board intended for active communication with and among the community.
Please feel free to ask your questions there, share valuable insights and give us feedback on our material. 

# 1. Usage

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

The project contains some jupyter notebooks, which were converted to python files
due to better handling in the repository.
These files end with `_nb.py` and can be converted back to a `.ipynb` file with
`jupytext`:

```shell
jupytext --to ipynb --execute <your_file>_nb.py
```

The `--execute` flag triggers executing every cell during conversion.
Alternatively, you can run the `_nb.py` files like every other python script.

# Contributing

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

To ensure code serialization and keeping the memory profile low, `.ipynb` are blacklisted
in this repository.
A notebook can be saved to the repo by converting it to a serializable format via
`jupytext`, preferably `py:percent`:

```shell
jupytext --to py:percent <notebook-to-convert>.ipynb
```

The result is a python file, which can be committed and later on be converted back to `.ipynb`.
A notebook-python file from jupytext shall carry the suffix `_nb.py`.
