"""Script for training Berttopic on custom dataset ."""
import os
import urllib.request
import zipfile

import pandas as pd
from bertopic import BERTopic

from bearml.topic_modeling.preprocessing import StringPreprocessor


def download_dataset(f_dir: str = 'data') -> None:
    """Download the Economist Historical Advertisements – Master dataset."""
    url = (
        'http://openbigdata.directory/wp-content/uploads/'
        'TheEconomistHistoricalArchives-MasterDataset.zip'
    )
    csv_name = 'The Economist Historical Advertisements – Master Dataset.csv'
    csv_path = os.path.join(f_dir, csv_name)
    zip_name = 'eha.zip'
    zip_path = os.path.join(f_dir, zip_name)

    # Skip download and extraction if file already exists.
    # This is a very rudimentary check and should not be done in a serious project.
    # The file is not checked for completeness. Use SHA or MD5 hashes otherwise.
    if os.path.exists(csv_path):
        return

    # Download the zip file.
    print('Downloading dataset ...')
    urllib.request.urlretrieve(url, zip_path)

    print('Extracting dataset ...')
    # Extract the dataset.
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(f_dir)


def main():
    """Run training."""
    # We will use the "The Economist Historical Advertisements – Master Dataset" for
    # Topic modeling. The following function downloads the dataset to `data` and
    # extracts the zip archive.
    # Also checkout:
    # https://www.berd-nfdi.de/2021/11/03/dataset-on-historical-advertisements-of-the-economist-released-now/ # noqa
    download_dataset()

    # Read csv file to memory as pandas dataframe
    print('Loading dataset ...')
    f_name = 'The Economist Historical Advertisements – Master Dataset.csv'
    f_path = os.path.join('data', f_name)
    df = pd.read_csv(f_path, header=0, dtype=str)

    print('Preprocessing dataset ...')
    # We extract the column containing the advertisement text
    docs = df['OCR_GoogleVision_original'].tolist()

    # Limit number of samples to save compute in this example.
    docs = docs[:10000]

    # Initialize preprocessing class
    # Spell correction is off for computational efficiency
    preprocessor = StringPreprocessor(correct_spelling=False)

    # Execute preprocessing steps
    docs = preprocessor.preprocess(docs)

    print('Analyzing topics ...')
    # Use a pre-trained BERTopic model for prediction
    # You can find more information about details of Berttopic in
    # https://github.com/MaartenGr/BERTopic
    topic_model = BERTopic(
        embedding_model='xlm-r-bert-base-nli-stsb-mean-tokens', verbose=True
    )
    topic_model.fit(docs)

    # View our top 5 identified topics!
    print(topic_model.get_topic_info().head(5))


if __name__ == '__main__':
    main()
