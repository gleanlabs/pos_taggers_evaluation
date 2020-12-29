import pandas as pd
import nltk
from tokenizer.treebankwordtokenizer_spacy_whitespace import RevisedTreeBankWordTokenizerVocab
import stanza
import en_core_web_sm
import os
import ast
from flair.models import SequenceTagger
from source.tokenizer_functions import _split_composite_pos_tokens
from flair.data import Sentence
import itertools
import json

THIS_FOLDER = "/Users/johanna/Desktop/pos_taggers_evaluation/pos_taggers_evaluation/"


def nltk_pos_fct(sent_tok: list):
    sent_tags = [nltk.pos_tag(sub_sent_tok) for sub_sent_tok in sent_tok]
    return [item for sublist in sent_tags for item in sublist]


def stanza_pos_fct(sent_tok: list):
    # uses batches
    nlp_stanza = stanza.Pipeline(lang='en', processors='tokenize,pos', tokenize_pretokenized=True)
    pos_batch = [[(word.text, word.xpos) for word in s.words] for s in nlp_stanza(sent_tok).sentences]
    return [item for sublist in pos_batch for item in sublist]


def spacy_pos_fct(sent_tok: list):
    nlp_spacy = en_core_web_sm.load()
    nlp_spacy.tokenizer = RevisedTreeBankWordTokenizerVocab(nlp_spacy.vocab)
    sent_tags = [[(word.text, word.pos_) for word in nlp_spacy(s)] for s in sent_tok]
    return [item for sublist in sent_tags for item in sublist]


def flair_pos_fct(sent_tok: list):
    # uses batches
    tagger = SequenceTagger.load('pos')
    sentences = [Sentence(i, use_tokenizer=False) for i in sent_tok]
    tagger.predict(sentences)
    sent_tags = [[(word.text, word.get_tag('pos').value) for word in sentence] for sentence in sentences]
    return [item for sublist in sent_tags for item in sublist]


def split_labels_articles_that_need_to(gt):
    dict_lab = {"Z": ["^", "T"], "S": ["N", "T"], "L": ["O", "V"], "M": ["^", "V"], "Y": ["X", "V"]}
    with open(os.path.join(THIS_FOLDER, 'source/utils/tokens_to_split.json')) as json_file:
        data = json.load(json_file)
    new_labels = []
    for gt_val in gt:
        if gt_val[1] in ["Z", "S", "L", "M", "Y"]:
            new_labels.append(dict_lab[gt_val[1]])
        elif gt_val[0] in data.keys():
            new_labels.append([gt_val[1], 'T'])
        else:
            new_labels.append([gt_val[1]])
    return list(itertools.chain.from_iterable(new_labels))


def article_gt(sent: list):
    df_pos = pd.read_csv(os.path.join(THIS_FOLDER, 'source/utils/sentences_to_GT_POS.csv'))
    gt = split_labels_articles_that_need_to(df_pos[df_pos.sentence == sent]['tagged_tokens_GT'].apply(
        lambda x: ast.literal_eval(x)).tolist()[0])
    sent_tok = _split_composite_pos_tokens(df_pos[df_pos.sentence == sent]['tagged_tokens_GT'].apply(
        lambda x: [i[0] for i in ast.literal_eval(x)]).tolist()[0])
    return [(sent_tok_val, gt_val) for gt_val, sent_tok_val in zip(gt, sent_tok)]
