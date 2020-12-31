"""
Module which tags POS given a list of sentences and a package
"""
from source.pos_taggers_functions import nltk_pos_fct, stanza_pos_fct, spacy_pos_fct, flair_pos_fct, article_gt
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
    if pos_tagger == 'article':
        return (article_gt(doc, os.path.join(THIS_FOLDER, 'source/eval/sentences_to_GT_POS_corrected.csv')))
    return res


def _read_tag_map():
    with open(os.path.join(THIS_FOLDER, 'source/utils/tag_map.json')) as json_file:
        data = json.load(json_file)
    return data


def map_results_to_universal_tags(raw_tokens: list, source: str):
    mapping = _read_tag_map()
    if source == 'nltk':
        dict_mapping = mapping['PTB-UNIV']
        results_mapped = [(tags[0], dict_mapping[tags[1]]) for tags in raw_tokens]
    if source == 'stanza':
        dict_mapping = mapping['PTB-UNIV']
        results_mapped = [(tags[0], dict_mapping[tags[1]]) for tags in raw_tokens]
    if source == 'spacy':
        dict_mapping = mapping['UNIV']
        results_mapped = [(tags[0], dict_mapping[tags[1]]) for tags in raw_tokens]
    if source == 'flair':
        dict_mapping = mapping['PTB-UNIV']
        results_mapped = [(tags[0], dict_mapping[tags[1]]) for tags in raw_tokens]
    if source == 'article':
        dict_mapping = mapping['ARTICLE-UNIV']
        results_mapped = [(tags[0], dict_mapping[tags[1]]) for tags in raw_tokens]
    return results_mapped
