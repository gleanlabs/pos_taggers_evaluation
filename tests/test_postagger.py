"""
A series of tests to check that pos tagging for each package occurs as expected
"""
from source.tag_pos import _pos_tag_sentence, _read_tag_map, map_results_to_universal_tags
import pytest
from source.tokenizer_functions import tokenize


@pytest.fixture
def documents():
    documents_list = [
        "Thats an approach I had not considered . I'd end up polling from the Javascript until something changes , but that might work .",
        "URL decoding in Javascript",
        "How to register a JavaScript callback in a Java Applet ?",
        "I would not disagree with this .",
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
        assert [i[0] for i in _pos_tag_sentence('article', doc)] == flatten_list_tokens


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
            'flair', doc)) ==   len(_pos_tag_sentence(
            'article', doc))


def test_each_token_has_a_tag(documents: list):
    """
    Check that each tag is a valid key in the tag_map
    """
    mappings = _read_tag_map()
    keys = list(mappings['UNIV'].keys()) + list(mappings['PTB-UNIV'].keys()) + list(mappings['ARTICLE-UNIV'].keys())
    for doc in documents:
        assert all(item in keys for item in [i[1] for i in _pos_tag_sentence('nltk', doc)]) == True
        assert all(item in keys for item in [i[1] for i in _pos_tag_sentence('stanza', doc)]) == True
        assert all(item in keys for item in [i[1] for i in _pos_tag_sentence('spacy', doc)]) == True
        assert all(item in keys for item in [i[1] for i in _pos_tag_sentence('flair', doc)]) == True
        assert all(item in keys for item in [i[1] for i in _pos_tag_sentence('article', doc)]) == True


def test_each_token_has_a_mapped_correct_tag(documents: list):
    """
    Check that each mapped tag is a valid value
    """
    mappings = _read_tag_map()
    values = list(mappings['UNIV'].values()) + list(mappings['PTB-UNIV'].values()) + list(
        mappings['ARTICLE-UNIV'].values())
    for doc in documents:
        assert all(
            item in values for item in
            [i[1] for i in map_results_to_universal_tags(_pos_tag_sentence('nltk', doc), 'nltk')]) == True
        assert all(item in values for item in [i[1] for i in
                                               map_results_to_universal_tags(_pos_tag_sentence('stanza', doc),
                                                                             'stanza')]) == True
        assert all(
            item in values for item in [i[1] for i in
                                        map_results_to_universal_tags(_pos_tag_sentence('spacy', doc),
                                                                      'spacy')]) == True
        assert all(
            item in values for item in
            [i[1] for i in map_results_to_universal_tags(_pos_tag_sentence('flair', doc), 'flair')]) == True
        assert all(
            item in values for item in
            [i[1] for i in map_results_to_universal_tags(_pos_tag_sentence('article', doc), 'article')]) == True
