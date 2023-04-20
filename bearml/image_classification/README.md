# Image Classification with PyTorch Lightning

In this example, we showcase how training a classifier works with pytorch lightning.

- `data.py` contains the data module.
- `model.py` contains the model module.
- `main.py` contains the run and trainer configuration

Here we rely mostly on standard methods. The `CIFAR10` dataset from `torchvision` is
used, and we utilize the `torch.hub` to download model definitions.

Call the main script to start training:

```shell
python bearml/image_classification/main.py
```

There a few command line options available, which can be displayed by adding the
`--help` flag to the above command.
