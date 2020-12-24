import pandas as pd
import nltk
from tokenizer.treebankwordtokenizer_spacy import RevisedTreeBankWordTokenizerVocab
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
    return nltk.pos_tag(sent_tok)


def stanza_pos_fct(sent_tok: list):
    nlp_stanza = stanza.Pipeline(lang='en', processors='tokenize,pos', tokenize_pretokenized=True)
    pos_batch = [[(word.text, word.xpos) for word in s.words] for s in nlp_stanza(sent_tok).sentences]
    return [item for sublist in pos_batch for item in sublist]


def spacy_pos_fct(sent_tok: list):
    nlp_spacy = en_core_web_sm.load()
    nlp_spacy.tokenizer = RevisedTreeBankWordTokenizerVocab(nlp_spacy.vocab)
    return [(word.text, word.pos_) for word in nlp_spacy(sent_tok)]


def flair_pos_fct(sent_tok: list):
    tagger = SequenceTagger.load('pos')
    sentences = Sentence(sent_tok, use_tokenizer=False)
    tagger.predict(sentences)
    return [(word.text, word.get_tag('pos').value) for word in sentences]

nlp_stanza = stanza.Pipeline(lang='en', processors='tokenize,pos', tokenize_pretokenized=True)
# print(nlp_stanza([['I', 'am'], ['I', 'am']]))
# print(list(np.concatenate(np.array([[(word.text, word.xpos) for word in s.words] for s in nlp_stanza([['I', 'am'], ['I', 'am']]).sentences]),
#                        axis=1)))

tagger = SequenceTagger.load('pos')
sentences = [Sentence(i, use_tokenizer=False) for i in [['I', 'am'], ['I', 'am']]]
tagger.predict(sentences)
print(sentences)
print(sentences[0][0].get_tag('pos'))
print([[word.get_tag('pos').value for word in sentences] for sentence in sentences])
print([[(word.text, word.get_tag('pos').value) for word in sentences] for sentence in sentences])
