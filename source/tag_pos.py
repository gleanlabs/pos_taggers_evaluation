"""
Module which tags POS given a list of sentences and a package
"""
from source.pos_taggers_functions import nltk_pos_fct, stanza_pos_fct, spacy_pos_fct, flair_pos_fct
import os
from source.tokenizer_functions import tokenize
import json

THIS_FOLDER = "/Users/johanna/Desktop/pos_taggers_evaluation/pos_taggers_evaluation/"


def _choose_package(package_name: str):
    return package_name


def _pos_tag_sentence(package_name: str, doc: str):
    pos_tagger = _choose_package(package_name)
    res = []
    if pos_tagger == 'nltk':
        return (nltk_pos_fct(tokenize(doc)))
    if pos_tagger == 'spacy':
        return (spacy_pos_fct(tokenize(doc)))
    if pos_tagger == 'stanza':
        return (stanza_pos_fct(tokenize(doc)))
    if pos_tagger == 'flair':
        return (flair_pos_fct(tokenize(doc)))
    return res


def _read_tag_map():
    with open(os.path.join(THIS_FOLDER, 'source/utils/tag_map.json')) as json_file:
        data = json.load(json_file)
    return data

def map_results_to_universal_tags(raw_tokens: list, source: str):
    mapping = _read_tag_map()
    results_mapped = []
    return results_mapped
