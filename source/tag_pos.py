"""
Module which tags POS given a list of sentences and a package
"""
from source.pos_taggers_functions import nltk_pos_fct, stanza_pos_fct, spacy_pos_fct, flair_pos_fct

from source.tokenizer_functions import tokenize


def _choose_package(package_name: str):
    return package_name


def _pos_tag_sentence(package_name: str, sent: str):
    pos_tagger = _choose_package(package_name)
    res = []
    if pos_tagger == 'nltk':
        for sent_tok in tokenize(sent):
            res.append([ nltk_pos_fct(tok) for tok in nltk_pos_fct(sent_tok)])
    if pos_tagger == 'spacy':
        for s in sent:
            res.append(spacy_pos_fct(tokenize(sent)))
    return res


def _pos_tag_batch(package_name: str, batch: list):  # check syntax for type hints
    pos_tagger = _choose_package(package_name)
    res = []
    if pos_tagger == 'stanza':
        for sent in batch:
            res.append(stanza_pos_fct(tokenize(sent)))
    if pos_tagger == 'flair':
        for sent in batch:
            res.append(flair_pos_fct(tokenize(sent)))
    return res

#
# def _read_tag_map(tag_path: str):
#     tag_map = ""
#     return tag_map
#
#
# def tag_pos(package_name: str, sentences: list):
#     if package_name in ['stanza', 'flair']:
#         result = [_pos_tag_batch(batch) for batch in sentences]
#         return [r for r in [res for res in result]]  # flatten list
#     else:
#         return [_pos_tag_sentence(sent) for sent in sentences]
#
#
# def map_results_to_universal_tags(raw_tokens: list[tuple], tag_map):
#     results_mapped = []
#     return results_mapped
