"""
Sanity checks that the analysis produce valid results
"""
import pytest
from source.tag_pos import _pos_tag_sentence, map_results_to_universal_tags
from source.tag_statistics import *


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
        for nltk_tag, stanza_tag, spacy_tag, flair_tag, article_tag in zip(
                map_results_to_universal_tags(_pos_tag_sentence('nltk', doc), 'nltk'),
                map_results_to_universal_tags(
                    _pos_tag_sentence('stanza', doc),
                    'stanza'),
                map_results_to_universal_tags(
                    _pos_tag_sentence('spacy', doc),
                    'spacy'),
                map_results_to_universal_tags(
                    _pos_tag_sentence('flair', doc), 'flair'),
                map_results_to_universal_tags(
                    _pos_tag_sentence('article', doc),
                    'article')):
            assert len([i[1] for i in [nltk_tag, stanza_tag, spacy_tag, flair_tag, article_tag]]) == 5
            assert len(list(set([i[0] for i in [nltk_tag, stanza_tag, spacy_tag, flair_tag, article_tag]]))) == 1


def test_token_majority(documents: list):
    """we check that we have the token with the majority votes is indeed the right one, and we check how many votes he gets
    """
    for doc in documents:
        for nltk_tag, stanza_tag, spacy_tag, flair_tag, article_tag in zip(
                map_results_to_universal_tags(_pos_tag_sentence('nltk', doc), 'nltk'),
                map_results_to_universal_tags(
                    _pos_tag_sentence('stanza', doc),
                    'stanza'),
                map_results_to_universal_tags(
                    _pos_tag_sentence('spacy', doc),
                    'spacy'),
                map_results_to_universal_tags(
                    _pos_tag_sentence('flair', doc), 'flair'),
                map_results_to_universal_tags(
                    _pos_tag_sentence('article', doc),
                    'article')):
            token_tags = [nltk_tag, stanza_tag, spacy_tag, flair_tag, article_tag]
            number_votes_majority_token = return_number_votes_majority_token(token_tags)
            assert number_votes_majority_token <= len(token_tags)
            assert number_votes_majority_token >= 1
            assert number_votes_majority_token == max([token_tags.count(i) for i in token_tags])


def test_unique_tokens_voted(documents: list):
    """we check that the number of unique tokens voted is the right number
    """
    for doc in documents:
        for nltk_tag, stanza_tag, spacy_tag, flair_tag, article_tag in zip(
                map_results_to_universal_tags(_pos_tag_sentence('nltk', doc), 'nltk'),
                map_results_to_universal_tags(
                    _pos_tag_sentence('stanza', doc),
                    'stanza'),
                map_results_to_universal_tags(
                    _pos_tag_sentence('spacy', doc),
                    'spacy'),
                map_results_to_universal_tags(
                    _pos_tag_sentence('flair', doc), 'flair'),
                map_results_to_universal_tags(
                    _pos_tag_sentence('article', doc),
                    'article')):
            token_tags = [nltk_tag, stanza_tag, spacy_tag, flair_tag, article_tag]
            nb_unique_tokens = return_unique_tokens(token_tags)
            number_votes_majority_token = return_number_votes_majority_token(token_tags)
            assert nb_unique_tokens <= len(token_tags)
            assert nb_unique_tokens >= 1
            assert nb_unique_tokens <= len(token_tags) - number_votes_majority_token + 1


def test_wether_majority_token_equals_gt(documents: list):
    """we check if the token voted in majority equals or not the GT
        """
    for doc in documents:
        for nltk_tag, stanza_tag, spacy_tag, flair_tag, article_tag in zip(
                map_results_to_universal_tags(_pos_tag_sentence('nltk', doc), 'nltk'),
                map_results_to_universal_tags(
                    _pos_tag_sentence('stanza', doc),
                    'stanza'),
                map_results_to_universal_tags(
                    _pos_tag_sentence('spacy', doc),
                    'spacy'),
                map_results_to_universal_tags(
                    _pos_tag_sentence('flair', doc), 'flair'),
                map_results_to_universal_tags(
                    _pos_tag_sentence('article', doc),
                    'article')):
            token_tags = [nltk_tag, stanza_tag, spacy_tag, flair_tag, article_tag]
            majority_token = return_majority_token(token_tags)
            bool = return_wether_majority_token_equals_gt(token_tags)
            assert bool in [True, False]
            assert bool == (majority_token == token_tags[-1])
