"""
Module for tokenize text prior to pos tagging
"""

import nltk
import itertools


def tokenize(doc: str):
    return [text.split(' ') for text in nltk.sent_tokenize(doc)]


def split_tok_articles_that_need_to(sent, gt):
    new_tok = []
    d = "'"
    for tok, gt_val in zip(sent.split(' '), gt):
        if "'" in tok and gt_val in ["Z", "S", "L", "M", "Y"]:
            new_tok.append([d + e if e != tok.split(d)[0] else e for e in tok.split(d)])
        else:
            new_tok.append([tok])
    return list(itertools.chain.from_iterable(new_tok))
