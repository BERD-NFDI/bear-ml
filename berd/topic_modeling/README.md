# Topic modeling with Berttopic

In this example, we showcase how to use preprocessing and use the results for a topic modeling task.


- `preprocessing.py` contains the preprocess module.
- `main.py` contains the run and trainer configuration

### Topic modeling

Use the Berttopic modeling for a custom dataset. Since the Berttopic has its own tokenizer, the tokenizer function in preprocess class is commented out.

### Preprocessing

You can find the preprocessing class in Topic modeling/Utils. To use this class:
```python
method = preprocess()
results = [method.preprocessing(text) for text in B]
```

The preprocessing class contains 7 features:

 - Remove website links
 - Remove newline (/n)
 - Remove punctuations
 - Remove digits
 - Remove stop words based on gensim library stop words list
 - change all words to lowercase
 - tokenize the text based on DistilBert Tokenizer ('distilbert-base-uncased')

 you can use each of this feature separately for example:

 ```python
method = preprocess()
results = [method.remove_punctuation(text) for text in B]
```
or you can use all of them together with:

```python
method = preprocess()
results = [method.preprocessing(text) for text in B]
```

Call the main script to start training:

```shell
python berd/topic_modeling/main.py
```

There a few command line options available, which can be displayed by adding the
`--help` flag to the above command.

## Acknowledgements

- [BERTTopic modeling](https://github.com/MaartenGr/BERTopic)
