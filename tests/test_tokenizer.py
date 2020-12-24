"""
Some functions that test the proper working of the tokenizer prior to pos tagging
"""
from source.tokenizer import tokenize
import pytest


@pytest.fixture
def sentences():
    sentences = [
        ""
    ]
    return sentences


@pytest.fixture
def tokens():
    expected = [
        ("", ""),
        ("", "")
    ]
    return expected


def test_correct_tokens(sentences: list[str], expected: list[tuple]):
    token_list = [tokenize(sent) for sent in sentences]
    assert len(token_list) == len(expected)  # the same number of sentences got tokenized
    for sent, exp in zip(sentences, expected):
        assert (len(sent) == len(exp))  # test that the expected number of tokens was produced for each sentence
        for actual, exp_token in zip(sent, exp):
            assert (actual == exp_token)  # test that each token is the same as the expected token
