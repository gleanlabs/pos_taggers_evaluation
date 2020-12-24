"""
Module for tokenize text prior to pos tagging
"""

import nltk
import itertools
import json

def tokenize(doc: str):
    return _split_composite_pos_tokens([text.split(' ') for text in nltk.sent_tokenize(doc)])


def _split_composite_pos_tokens(sent_tok):
    with open('utils/tokens_to_split.json') as json_file:
        data = json.load(json_file)
    new_tok = []
    for tok, gt_val in zip(sent_tok):
        if tok in data.keys():
            new_tok.append(data[tok])
        else:
            new_tok.append([tok])
    return list(itertools.chain.from_iterable(new_tok))
