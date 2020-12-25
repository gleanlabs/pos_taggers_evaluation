"""
A series of tests to check that pos tagging for each package occurs as expected
"""
from source.tag_pos import _pos_tag_sentence
import pytest
from source.tokenizer_functions import tokenize


@pytest.fixture
def documents():
    documents_list = [
        "So the whole TCP/ IP checksum thing isn't working ... I'm thinking that anything corrupted in transit is going to get rejected at a much lower layer than the application level .",
        "I am coding in python.",
        "I am coding in python .",
        "Using JavaScript's escape / unescape function is almost always the wrong thing , it is incompatible with URL-encoding or any other standard encoding on the web . Non-ASCII characters are treated unexpectedly as well as spaces , and older browsers don't necessarily have the same behaviour . As mentioned by roenving , the method you want is decodeURIComponent() .",
    ]
    return documents_list


def test_tokens_for_each_packages(documents: list):
    """
    The idea is to confirm that the document that is tagged is the same as the original tokenized document
    """
    # tokens are the same
    for doc in documents:
        flatten_list_tokens = [item for sublist in tokenize(doc) for item in sublist]
        assert [i[0] for i in _pos_tag_sentence('nltk', doc)] == flatten_list_tokens
        assert [i[0] for i in _pos_tag_sentence('stanza', doc)] == flatten_list_tokens
        assert [i[0] for i in _pos_tag_sentence('spacy', doc)] == flatten_list_tokens
        assert [i[0] for i in _pos_tag_sentence('flair', doc)] == flatten_list_tokens


def test_each_package_returns_same_number_results(documents: list):
    """
    Some packages work on batches and others on individual sentences, make sure the resulted tagging has the correct number
    of tokens per document and that the tokens are the same
    """
    # tokens are the same
    for doc in documents:
        assert [i[0] for i in _pos_tag_sentence('nltk', doc)] == [i[0] for i in _pos_tag_sentence('stanza', doc)] == [
            i[0] for i in _pos_tag_sentence('spacy', doc)] == [i[0] for i in _pos_tag_sentence('flair', doc)]
    # same number of tokens
    for doc in documents:
        assert len(_pos_tag_sentence('nltk', doc)) == len(_pos_tag_sentence('stanza', doc)) == len(
            _pos_tag_sentence('spacy',
                              doc)) == len(_pos_tag_sentence(
            'flair', doc))


def test_each_token_has_a_tag():
    """
    Check that each tag is a valid key in the tag_map
    """
