"""Script for training Berttopic on custom dataset ."""

import pandas as pd
from bertopic import BERTopic

from berd.topic_modeling.preprocessing import preprocess


def main():
    """Run training."""
    # You need to download dataset from
    # https://www.berd-nfdi.de/2021/11/03/dataset-on-historical-advertisements-of-the-economist-released-now/ # noqa
    # Read csv file and convert to panda dataframe
    data = pd.read_csv(
        'Data/The Economist Historical Advertisements – Master Dataset.csv',
        header=0,
        dtype=str,
    )
    x = data.values.tolist()
    # Change dataset to string
    x_string = [str(a) for a in x]
    # Choose which part of  dataset you want to for Berttopic
    selection = x_string[:100]
    # Define preprocess class
    method = preprocess()
    # Apply preprocessing steps except tokenization
    # since Berttopic has its own tokenization
    results = [method.preprocessing(text) for text in selection]
    # Use pretrained Berttopic model for prediction
    # You can find more information about details of Berttopic in
    # https://github.com/MaartenGr/BERTopic # noqa
    topic_model = BERTopic(embedding_model='xlm-r-bert-base-nli-stsb-mean-tokens').fit(
        results
    )
    print(topic_model.get_topic_info().head(5))


if __name__ == '__main__':
    main()
