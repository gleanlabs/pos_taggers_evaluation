import pandas as pd
import nltk
from tokenizer.treebankwordtokenizer_spacy_whitespace import RevisedTreeBankWordTokenizerVocab
import stanza
import en_core_web_sm
import spacy
import ast
from flair.models import SequenceTagger
from flair.data import Sentence
from nltk.tokenize import word_tokenize
import numpy as np
import itertools


def nltk_pos_fct(sent_tok: list):
    sent_tags = [nltk.pos_tag(sub_sent_tok) for sub_sent_tok in sent_tok]
    return [item for sublist in sent_tags for item in sublist]


def stanza_pos_fct(sent_tok: list):
    #uses batches
    nlp_stanza = stanza.Pipeline(lang='en', processors='tokenize,pos', tokenize_pretokenized=True)
    pos_batch = [[(word.text, word.xpos) for word in s.words] for s in nlp_stanza(sent_tok).sentences]
    return [item for sublist in pos_batch for item in sublist]


def spacy_pos_fct(sent_tok: list):
    nlp_spacy = en_core_web_sm.load()
    nlp_spacy.tokenizer = RevisedTreeBankWordTokenizerVocab(nlp_spacy.vocab)
    sent_tags = [[(word.text, word.pos_) for word in nlp_spacy(s)] for s in sent_tok]
    return [item for sublist in sent_tags for item in sublist]


def flair_pos_fct(sent_tok: list):
    #uses batches
    tagger = SequenceTagger.load('pos')
    sentences = [Sentence(i, use_tokenizer=False) for i in sent_tok]
    tagger.predict(sentences)
    sent_tags = [[(word.text, word.get_tag('pos').value) for word in sentence] for sentence in sentences]
    return [item for sublist in sent_tags for item in sublist]
