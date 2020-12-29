"""
A series of tests to check that pos tagging for each package occurs as expected
"""
from source.tag_pos import _pos_tag_sentence, _read_tag_map, map_results_to_universal_tags
import pytest
from source.tokenizer_functions import tokenize

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


@pytest.fixture
def documents_pos_tagging_article():
    documents_list = [
        "Using JavaScript's escape / unescape function is almost always the wrong thing , it is incompatible with URL-encoding or any other "
        "standard encoding on the web . Non-ASCII characters are treated unexpectedly as well as spaces , and older browsers don't necessarily"
        " have the same behaviour . As mentioned by roenving , the method you want is decodeURIComponent() . This is a newer addition which you "
        "won't find on IE 5.0 , so if you need to support that browser ( let's hope not , nowadays ! ) you'd need to implement the function yourself ."
        " And for non-ASCII characters that means you need to implement a UTF-8 encoder . Code is available if you need it ."
    ]
    return documents_list


def test_pos_tagging_article(documents_pos_tagging_article: list):
    """
    The idea is to confirm that the document that is tagged is the same as the original tokenized document
    """
    # tokens are the same
    for doc in documents_pos_tagging_article:
        assert [i[1] for i in _pos_tag_sentence('article', doc)][1] == '^'


def test_tokens_for_each_packages(documents: list):
    """
    The idea is to confirm that the document that is tagged is the same as the original tokenized document
    """
    # tokens are the same
    for doc in documents:
        flatten_list_tokens = [item for sublist in tokenize(doc) for item in sublist]
        for lib in LIST_PACKAGES:
            assert [i[0] for i in _pos_tag_sentence(lib, doc)] == flatten_list_tokens


def test_each_package_returns_same_number_results(documents: list):
    """
    Some packages work on batches and others on individual sentences, make sure the resulted tagging has the correct number
    of tokens per document and that the tokens are the same
    """
    # tokens are the same
    for doc in documents:
        tokens_ref = [i[0] for i in _pos_tag_sentence(LIST_PACKAGES[0], doc)]
        for lib in LIST_PACKAGES:
            assert tokens_ref == [i[0] for i in _pos_tag_sentence(lib, doc)]
            # same number of tokens
    for doc in documents:
        length_ref = len(_pos_tag_sentence(LIST_PACKAGES[0], doc))
        for lib in LIST_PACKAGES:
            assert length_ref == len(_pos_tag_sentence(lib, doc))


def test_each_token_has_a_tag(documents: list):
    """
    Check that each tag is a valid key in the tag_map
    """
    mappings = _read_tag_map()
    keys = list(mappings['UNIV'].keys()) + list(mappings['PTB-UNIV'].keys()) + list(mappings['ARTICLE-UNIV'].keys())
    for doc in documents:
        for lib in LIST_PACKAGES:
            assert all(item in keys for item in [i[1] for i in _pos_tag_sentence(lib, doc)]) == True


def test_each_token_has_a_mapped_correct_tag(documents: list):
    """
    Check that each mapped tag is a valid value
    """
    mappings = _read_tag_map()
    values = list(mappings['UNIV'].values()) + list(mappings['PTB-UNIV'].values()) + list(
        mappings['ARTICLE-UNIV'].values())
    for doc in documents:
        for lib in LIST_PACKAGES:
            assert all(
                item in values for item in
                [i[1] for i in map_results_to_universal_tags(_pos_tag_sentence(lib, doc), lib)]) == True
