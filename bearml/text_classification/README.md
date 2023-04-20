# Text Classification with PyTorch Lightning

In this example, we showcase how training a text classifier works with pytorch lightning.

- `data.py` contains the data module.
- `model.py` contains the model module.
- `main.py` contains the run and trainer configuration

Here we rely mostly on standard methods. We used [Toxic comment dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) from kaggle toxic comment challenge which contains toxic comments.

We rely on pretrained BERT as a language model that provides text embeddings for down stream task (here text classification). To get best results, you should unfreeze the weights in BERT model but if you have limited computation power you can freeze weights in pretrained model.
Call the main script to start training:

```shell
python bearml/text_classification/main.py
```

There are a few command line options available, which can be displayed by adding the
`--help` flag to the above command.

## Acknowledgements

- [Toxic comment dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- [Text classification](https://curiousily.com/posts/multi-label-text-classification-with-bert-and-pytorch-lightning/)
