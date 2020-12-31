"""
Sanity checks that the analysis produce valid results
"""
import pytest
from source.tag_pos import _pos_tag_sentence, map_results_to_universal_tags
from source.tag_statistics import *

LIST_PACKAGES = ['nltk', 'stanza', 'spacy', 'flair', 'article']


@pytest.fixture
def documents():
    documents_list = [
        "Thats an approach I had not considered . I'd end up polling from the Javascript until something changes , but that might work .",
        "URL decoding in Javascript",
        "How to register a JavaScript callback in a Java Applet ?",
        "I would not disagree with this .",
    ]
    return documents_list


def test_nb_votes(documents: list):
    """we check that we have the right number of votes for a given token
    """
    for doc in documents:
        iterables = [map_results_to_universal_tags(_pos_tag_sentence(lib, doc), lib) for lib in LIST_PACKAGES]
        for list_token_tags in zip(*iterables):
            assert len([i[1] for i in list_token_tags]) == len(LIST_PACKAGES)
            assert len(list(set([i[0] for i in list_token_tags]))) == 1


def test_token_majority(documents: list):
    """we check that we have the token with the majority votes is indeed the one having most votes, and we check how many votes he gets
    """
    for doc in documents:
        iterables = [map_results_to_universal_tags(_pos_tag_sentence(lib, doc), lib) for lib in LIST_PACKAGES]
        for list_token_tags in zip(*iterables):
            number_votes_majority_token = return_number_votes_majority_token(list_token_tags)
            assert number_votes_majority_token <= len(list_token_tags)
            assert number_votes_majority_token >= 1
            assert number_votes_majority_token == max([list_token_tags.count(i) for i in list_token_tags])


def test_unique_tokens_voted(documents: list):
    """we check that the number of unique tokens voted is the right number
    """
    for doc in documents:
        iterables = [map_results_to_universal_tags(_pos_tag_sentence(lib, doc), lib) for lib in LIST_PACKAGES]
        for list_token_tags in zip(*iterables):
            nb_unique_tokens = return_unique_tokens(list_token_tags)
            number_votes_majority_token = return_number_votes_majority_token(list_token_tags)
            assert nb_unique_tokens <= len(list_token_tags)
            assert nb_unique_tokens >= 1
            assert nb_unique_tokens <= len(list_token_tags) - number_votes_majority_token + 1


def test_wether_majority_token_equals_gt(documents: list):
    """we check whether the comparison between majority and GT is correct
        """
    for doc in documents:
        iterables = [map_results_to_universal_tags(_pos_tag_sentence(lib, doc), lib) for lib in LIST_PACKAGES]
        for list_token_tags in zip(*iterables):
            majority_token = return_majority_token(list_token_tags)
            bool = return_wether_majority_token_equals_gt(list_token_tags)
            assert bool in [True, False]
            assert bool == (majority_token == list_token_tags[-1])
