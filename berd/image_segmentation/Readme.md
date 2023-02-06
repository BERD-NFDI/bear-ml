## Details how to run/modify segmentation code



### Dataset
In this project, a simple dataset with one class for segmenttion is used.

### Modify  the pretrained model

You can choose other pretrained model and use them. Also you can choose whether the weights are pretrained or not by choosing "None" instead of "imagenet" in the following code in main.py:
```python
model = segModel("FPN", "resnet34", in_channels=3, out_classes=1, lr=opt.lr, wd=opt.d)
```
There are other models with different backbone available in the SMP website. For more information check:
https://smp.readthedocs.io/en/latest/index.html

### Hyperparameter optimization

If you want to use custom dataset, you need to modify the hyperparameters. The important hyperparameters to modify are:

- Learning-rate (--lr)
- Batch-size (--batch_size)
- Weight decay (--wd)

To modify the learning-rate through training different schedulers are available, you can choose one in segmentation.py:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, min_lr=1e-5)
```
### Overfitting

To solve the overfitting problem, you can try the follwoing options:


- Early stopping: already implemented in pytorch-lightning callbacks
- Add transformation/augmentation to your data:
- Change which part of the model should be frozen
```python
for param in self.model.parameters():
    if model_w_f:
        param.requires_grad = False
    else:
        param.requires_grad = True
for param in self.model.segmentation_head.parameters():
    if head_w_f:
        param.requires_grad = False
    else:
        param.requires_grad = True
for param in self.model.decoder.parameters():
    if decoder_w_f:
        param.requires_grad = False
    else:
        param.requires_grad = True
 ```
- Increase the weight-decay
- Increase number of parameters in your model by adding more layers
