# Topic modeling with BERTopic

In this example, we showcase how to use preprocessing and use the results for a topic modeling task.


- `preprocessing.py` contains the preprocess module.
- `main.py` contains the run and trainer configuration

### Preprocessing

You can find the preprocessing class in `berd/topic_modeling/preprocessing.py`.
To use this class:
```python
from berd.topic_modeling.preprocessing import StringPreprocessor

docs = ['My dataset', 'with a list of strings']

preprocessor = StringPreprocessor()
docs = preprocessor.preprocess(docs)
```

The preprocessing class contains 7 features:

 - Remove website links
 - Remove newline (/n)
 - Remove punctuations
 - Remove digits
 - Remove stop words based on gensim library stop words list
 - change all words to lowercase

 you can use each of these features as independent functions:

 ```python
docs = [StringPreprocessor.remove_punctuation(text) for text in docs]
```
or you can use all of them together with:

```python
preprocessor = StringPreprocessor()
docs = preprocessor.preprocess(docs)
```

Call the main script to get embeddings for your documents:

```shell
python berd/topic_modeling/main.py
```

## Acknowledgements

- [BERTTopic modeling](https://github.com/MaartenGr/BERTopic)
