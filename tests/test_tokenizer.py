"""
Some functions that test the proper working of the tokenizer prior to pos tagging
"""
from source.tokenizer import tokenize
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


@pytest.fixture
def tokens():
    expected = [
        [['So', 'the', 'whole', 'TCP/', 'IP', 'checksum', 'thing', "isn't", 'working', '...'],
         ["I'm", 'thinking', 'that', 'anything', 'corrupted', 'in', 'transit', 'is', 'going', 'to', 'get', 'rejected',
          'at', 'a', 'much', 'lower', 'layer', 'than', 'the', 'application', 'level', '.']],
        [['I', 'am', 'coding', 'in', 'python.']],
        [['I', 'am', 'coding', 'in', 'python', '.']],
        [['Using', "JavaScript's", 'escape', '/', 'unescape', 'function', 'is', 'almost', 'always', 'the', 'wrong',
          'thing', ',', 'it', 'is', 'incompatible', 'with', 'URL-encoding', 'or', 'any', 'other', 'standard',
          'encoding', 'on', 'the', 'web', '.'],
         ['Non-ASCII', 'characters', 'are', 'treated', 'unexpectedly', 'as', 'well', 'as', 'spaces', ',', 'and',
          'older', 'browsers', "don't", 'necessarily', 'have', 'the', 'same', 'behaviour', '.'],
         ['As', 'mentioned', 'by', 'roenving', ',', 'the', 'method', 'you', 'want', 'is', 'decodeURIComponent()', '.']]
    ]
    return expected


def test_correct_tokens(sentences: list, tokens: list):
    token_list = [tokenize(sent) for sent in sentences]
    assert len(token_list) == len(tokens)  # the same number of sentences got tokenized
    for sent, exp in zip(token_list, tokens):
        assert (len(sent) == len(exp))  # test that the expected number of tokens was produced for each sentence
        for actual, exp_token in zip(sent, exp):
            assert (actual == exp_token)  # test that each token is the same as the expected token
