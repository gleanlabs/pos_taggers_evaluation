"""
A series of tests to check that pos tagging for each package occurs as expected
"""
from source.tag_pos import _pos_tag_sentence
import pytest


@pytest.fixture
def sentences():
    sentences = [
        "So the whole TCP/ IP checksum thing isn't working ... I'm thinking that anything corrupted in transit is going to get rejected at a much lower layer than the application level .",
        "I am coding in python.",
        "I am coding in python .",
        "Using JavaScript's escape / unescape function is almost always the wrong thing , it is incompatible with URL-encoding or any other standard encoding on the web . Non-ASCII characters are treated unexpectedly as well as spaces , and older browsers don't necessarily have the same behaviour . As mentioned by roenving , the method you want is decodeURIComponent() .",
    ]
    return sentences


def test_tokens_for_each_packages(sentences: list):
    """
    The idea is to confirm that the string that is tagged is the same as original tokens and same number of tokens
    """
    for sent in sentences:
        assert _pos_tag_sentence('nltk', sent) == _pos_tag_sentence('stanza', sent) == _pos_tag_sentence('spacy',
                                                                                                         sent) == _pos_tag_sentence(
            'flair', sent)


def test_each_package_returns_same_number_results():
    """
    Some packages work on batches and others on individual sentences, make sure the result has the correct number
    of tokens per sentence and correct of number of sentences
    """


def test_each_token_has_a_tag():
    """
    Check that each tag is a valid key in the tag_map
    """

print( _pos_tag_sentence('nltk',  "I am coding in python."))
print( _pos_tag_sentence('nltk',  "I am coding in python."))
print(_pos_tag_sentence('stanza', "I am coding in python."))
print(_pos_tag_sentence('spacy',"I am coding in python."))
print(_pos_tag_sentence('flair',"I am coding in python."))
