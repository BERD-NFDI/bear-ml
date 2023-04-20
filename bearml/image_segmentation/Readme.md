## Details how to run/modify the code for image segmentation

The backbone of this showcase is based on the `Segmentation Models PyTorch` package,
which provides a library with standard implementations and API of common segmentation
algorithms.

### Dataset
In this project, a simple dataset with one class for segmentation is used.

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
