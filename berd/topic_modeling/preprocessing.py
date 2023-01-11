"""Module class for a custom text preprocessing class."""

import logging
import re
import string

# load()
from autocorrect import Speller
from gensim.parsing.preprocessing import remove_stopwords

# from wordsegment import load, segment
from transformers import DistilBertTokenizer


class preprocess:
    """preprocess module."""

    def __init__(self):
        """Initialize the tokenizer model."""
        super(preprocess, self).__init__()
        # you can use tokenization if it is necessary.
        # There are different tokenizer available on hugginface.
        # We used Disitilbert tokenizer.
        # You can find more information about avialable pretrained tokenizer in
        # https://huggingface.co/docs/transformers/tokenizer_summary #qos
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def preprocessing(self, text):
        """Choose which preprocess function you want to be applied."""
        logging.info('In preprocess function.')
        pre_text = self.remove_website_links(text)
        pre_text = self.remove_newline(pre_text)
        pre_text = self.remove_punctuation(pre_text)
        pre_text = self.remove_digits(pre_text)
        pre_text = self.remove_stop_words(pre_text)
        pre_text = self.lowercase(pre_text)
        # pre_text = self.tokenize(pre_text)
        return pre_text

    def lowercase(self, text):
        """Convert dataframe to lowercase."""
        logging.info('Convert dataframe to lowercase')
        pre_text = text.lower()
        return pre_text

    def remove_newline(self, text):
        """Remove newlines from dataframe."""
        logging.info('Remove newlines from dataframe')
        pre_text = re.sub('\n', ' ', text)
        return pre_text

    def remove_punctuation(self, text):
        """Remove special characters."""
        logging.info('Remove special characters')
        pre_text = text.translate(str.maketrans('', '', string.punctuation))
        # punctuations = '''!()-![]{};:+'"\,<>./?@#$%^&*_~'''
        # pre_text = ' '.join([i for i in text if not i in punctuations])
        # pre_text = ' '.join(segment(pre_text))
        spell = Speller()
        pre_text = ' '.join([spell(w) for w in pre_text.split()])
        return pre_text

    def remove_digits(self, text):
        """Remove numbers from dataframe."""
        logging.info('Remove numbers from dataframe')
        pre_text = re.sub(r'\d', '', text)
        return pre_text

    def remove_stop_words(self, text):
        """Remove stop words from dataframe."""
        logging.info('Remove stop words from dataframe')
        pre_text = remove_stopwords(text)
        return pre_text

    def remove_website_links(self, text):
        """Remove website links from dataframe."""
        logging.info('Remove website links from dataframe')
        pre_text = re.sub(
            r'https?:\/\/(www\.)?[-a-zA-Z0–9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0–9@:%_\+.~#?&//=]*)',
            ' ',
            text,
            flags=re.MULTILINE,
        )

        pre_text = re.sub(
            r'[-a-zA-Z0–9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0–9@:%_\+.~#?&//=]*)',
            ' ',
            pre_text,
            flags=re.MULTILINE,
        )

        return pre_text

    def tokenize(self, text):
        """Apply tokenization."""
        logging.info('Apply tokenization')
        return self.tokenizer(text, truncation=True)
