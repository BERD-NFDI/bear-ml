# Topic modeling with BERTopic

In this example, we showcase how to use preprocessing and use the results for a topic modeling task.

## Table of contents
* [Background](#background)
* [Dataset](#dataset)
* [Code Structure](#code-structure)
* [Acknowledgements](#Acknowledgements)

## Background

### Topic modeling

Topic modeling is a type of unsupervised machine learning technique that is used to identify topics or themes that are
present in a collection of text documents. The goal of topic modeling is to automatically discover the underlying
semantic structure of the text data, and to group similar words and documents together based on their content.

Topic modeling has many applications, such as content analysis, information retrieval, and document clustering. It can
be used to gain insights into large collections of text data, and to identify patterns and trends that might not be
immediately apparent from manual inspection of the data.

### BERTopic

BERTopic is a topic modeling algorithm that uses the pre-trained BERT (Bidirectional Encoder Representations from
Transformers) language model to identify topics in a collection of text documents. BERTopic is based on the same
principles as traditional topic modeling algorithms such as Latent Dirichlet Allocation (LDA), but it uses BERT to
generate document embeddings, which are then clustered to identify topics.

The key idea behind BERTopic is to use BERT to encode each document in the corpus into a high-dimensional vector, which
captures its semantic meaning. This is done by extracting the contextual embeddings of each word in the document using
BERT, and then aggregating them into a single vector representation for the document.

Once the document embeddings have been generated, BERTopic uses a clustering algorithm to group similar documents
together, based on their semantic similarity. The clustering algorithm used by BERTopic is Hierarchical Density-Based
Spatial Clustering of Applications with Noise (HDBSCAN), which is a fast and efficient algorithm for clustering
high-dimensional data.

The output of BERTopic is a set of topics, where each topic is represented by a list of the most representative words
and a set of documents that belong to the topic. BERTopic also provides a variety of tools for visualizing the topics
and exploring the relationships between them.

One of the key advantages of BERTopic over traditional topic modeling algorithms is its ability to handle large and
complex text datasets, where traditional algorithms may struggle due to their assumptions about the underlying
distribution of the data. BERTopic has been shown to achieve state-of-the-art performance on a wide range of text
datasets, and has become a popular tool in the natural language processing (NLP) community.

### Preprocessing

Text preprocessing is the process of cleaning and transforming raw text data into a format that is suitable for analysis
or machine learning. Text data often contains noise, irregularities, and inconsistencies that can interfere with
downstream tasks, such as text classification, information retrieval, or language modeling. Text preprocessing aims to
remove these obstacles and to produce a clean and standardized representation of the text data.

There are several steps involved in text preprocessing, including:

- Tokenization: Breaking up the text into individual words or tokens.
- Lowercasing: Converting all text to lowercase to reduce the vocabulary size and to treat words with different cases as the same.
- Stop word removal: Removing common words such as "the", "and", and "is" that do not carry much meaning and can be safely ignored.
- Stemming or Lemmatization: Reducing words to their base form, such as converting "running" to "run" or "cars" to "car". This helps to reduce the vocabulary size and to group related words together.
- Removing special characters, punctuation, and numbers: Removing non-alphabetic characters that do not add much value to the text data.
- Spell checking and correction: Correcting common spelling errors and typos to improve the accuracy of downstream tasks.

## Dataset

We use [The Economist Historical Advertisements – Master Dataset](https://www.berd-nfdi.de/2021/11/03/dataset-on-historical-advertisements-of-the-economist-released-now/).
The Economist Historical Advertisements – Master Dataset is a collection of historical advertisements that were
published in The Economist magazine from 1843 to 2015. The dataset was compiled by a team of researchers at The
University of Tokyo and is publicly available for research purposes.

The dataset contains over 800,000 advertisements, covering a wide range of products and services, such as automobiles,
food, fashion, travel, and technology. The advertisements are presented in their original format, including text, images,
and layout.

The purpose of the dataset is to provide researchers with a valuable resource for studying the history of advertising,
consumer culture, and social change over the past 170 years. The dataset can be used for a wide range of research
projects, such as analyzing trends in advertising language, studying the evolution of consumer behavior, and exploring
the impact of advertising on society.

## Code Structure

This example contains two files. One is to showcase how to use `BERTTopic` and second is a customizable preprocessing for
text that contains different preprocessing steps.

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
