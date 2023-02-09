"""Module class for a custom text preprocessing class."""

import re
import string
from functools import reduce
from typing import List

from autocorrect import Speller
from gensim.parsing.preprocessing import remove_stopwords
from tqdm import tqdm


class StringPreprocessor:
    """Preprocessing module."""

    def __init__(self, **kwargs):
        """Initialize the StringPreprocessor."""
        super().__init__()

        # Choose the preprocessing functions that should be applied
        funcs = []
        if kwargs.get('remove_website_links', True):
            funcs.append(StringPreprocessor.remove_website_links)
        if kwargs.get('remove_newline', True):
            funcs.append(StringPreprocessor.remove_newline)
        if kwargs.get('remove_punctuation', True):
            funcs.append(StringPreprocessor.remove_punctuation)
        if kwargs.get('correct_spelling', True):
            funcs.append(StringPreprocessor.correct_spelling)
        if kwargs.get('remove_digits', True):
            funcs.append(StringPreprocessor.remove_digits)
        if kwargs.get('remove_stop_words', True):
            funcs.append(StringPreprocessor.remove_stop_words)
        if kwargs.get('convert_to_lowercase', True):
            funcs.append(StringPreprocessor.convert_to_lowercase)

        self.preprocess_funcs = tuple(funcs)

    def _exec_preprocess_funcs(self, s: str) -> str:
        """Execute all preprocessing functions in a chain."""
        return reduce(lambda arg, f: f(arg), self.preprocess_funcs, s)

    def preprocess(self, docs: List[str]) -> List[str]:
        """Preprocess a list of strings."""
        # This function works only on one core.
        # For larger datasets a look into the `multiprocessing` library helps.
        return [self._exec_preprocess_funcs(s) for s in tqdm(docs)]

    @staticmethod
    def convert_to_lowercase(s: str) -> str:
        """Convert string to lowercase."""
        return s.lower()

    @staticmethod
    def remove_newline(s: str) -> str:
        """Remove newlines from string."""
        return re.sub('\n', ' ', s)

    @staticmethod
    def remove_punctuation(s: str) -> str:
        """Remove special characters."""
        return s.translate(str.maketrans('', '', string.punctuation))

    @staticmethod
    def correct_spelling(s: str) -> str:
        """Correct spelling mistakes with autocorrect."""
        spell = Speller()
        return ' '.join([spell(w) for w in s.split(' ')])

    @staticmethod
    def remove_digits(s: str) -> str:
        """Remove numbers from string."""
        return re.sub(r'\d', '', s)

    @staticmethod
    def remove_stop_words(s: str) -> str:
        """Remove stop words from string."""
        return remove_stopwords(s)

    @staticmethod
    def remove_website_links(s: str) -> str:
        """Remove website links from string."""
        s = re.sub(
            r'https?:\/\/(www\.)?[-a-zA-Z0–9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0–9@:%_\+.~#?&//=]*)',  # noqa
            ' ',
            s,
            flags=re.MULTILINE,
        )

        s = re.sub(
            r'[-a-zA-Z0–9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0–9@:%_\+.~#?&//=]*)',  # noqa
            ' ',
            s,
            flags=re.MULTILINE,
        )

        return s
