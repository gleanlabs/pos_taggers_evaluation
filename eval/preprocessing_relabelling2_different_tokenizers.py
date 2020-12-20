import pandas as pd
import nltk
from tokenizer.treebankwordtokenizer_spacy import RevisedTreeBankWordTokenizerVocab
import stanza
import en_core_web_sm
# stanza.download('en')
import spacy
import ast
from flair.models import SequenceTagger
from flair.data import Sentence
from textblob import TextBlob
from textblob.base import BaseTokenizer
from nltk.tokenize import word_tokenize

spacy.load('en_core_web_sm')
import numpy as np
import itertools


def split_tok_articles_that_need_to(sent, gt):
    new_tok = []
    d = "'"
    for tok, gt_val in zip(sent.split(' '), gt):
        if "'" in tok and gt_val in ["Z", "S", "L", "M", "Y"]:
            new_tok.append([d + e if e != tok.split(d)[0] else e for e in tok.split(d)])
        else:
            new_tok.append([tok])
    return list(itertools.chain.from_iterable(new_tok))


def split_labels_articles_that_need_to(sent, gt):
    dict_lab = {"Z": ["^", "T"], "S": ["N", "T"], "L": ["O", "V"], "M": ["^", "V"], "Y": ["X", "V"]}
    new_labels = []
    d = "'"
    for tok, gt_val in zip(sent.split(' '), gt):
        if "'" in tok and gt_val in ["Z", "S", "L", "M", "Y"]:
            new_labels.append(dict_lab[gt_val])
        else:
            new_labels.append([gt_val])
    return list(itertools.chain.from_iterable(new_labels))


def get_index(sent, sent_tok):
    return [sent.index(tok) for tok in sent_tok]


df_pos = pd.read_csv('sentences_to_GT_POS.csv')
df_pos['GT'] = df_pos['tagged_tokens_GT'].apply(lambda x: [i[1] for i in ast.literal_eval(x)])
df_pos['sentence_tok'] = df_pos[['sentence', 'GT']].apply(lambda x: split_tok_articles_that_need_to(x[0], x[1]), axis=1)
df_pos['GT'] = df_pos[['sentence', 'GT']].apply(lambda x: split_labels_articles_that_need_to(x[0], x[1]), axis=1)
df_pos['GT_index'] = df_pos[['sentence', 'sentence_tok']].apply(lambda x: get_index(x[0], x[1]), axis=1)

ARTICLE_TO_UNIVERSAL_MAP = dict([
    ("&", "CONJ"), ("$", "NUM"), ("D", "DET"), ("P", "SCONJ/ADP"), ("A", "ADJ"), ("N", "NOUN"),
    ("O", "PRON"), ("R", "ADV"), ("V", "VERB"), ("^", "PROPN"), ("G", "SYM"), ("!", "INTJ"), ("T", "PART"),
    ("X", "DET"), (",", "PUNCT")
])

PENN_TREEBANK_TO_UNIVERSAL_MAP = dict([
    ("CC", "CCONJ"), ("CD", "NUM"), ("DT", "DET"), ("PDT", "DET"), ("FW", "X"), ("IN", "SCONJ/ADP"), ("JJ", "ADJ"),
    ("JJR", "ADJ"),
    ("JJS", "ADJ"), ("NN", "NOUN"), ("NNS", "NOUN"), ("MD", "VERB"),
    ("PRT", "PRT"), ("PRP", "PRON"), ("RB", "ADV"), ("RBR", "ADV"), ("RBS", "ADV"), ("WRB", "ADV"), ("VB", "VERB"),
    ("VBD", "VERB"),
    ("VBG", "VERB"), ("VBN", "VERB"), ("VBP", "VERB"), ("VBZ", "VERB"), ("NNP", "PROPN"),
    ("NNPS", "PROPN"), ("SYM", "SYM"), ("RP", "SCONJ"),
    (".", "PUNCT"), ("UH", "INTJ"), ("POS", "PRON"), ("PRP$", "PRON"), ("WDT", "DET"),
    ("WP", "PRON"), ("TO", "SCONJ/ADP"), ("-LRB-", "PUNCT"), ("RRB-", "PUNCT"), ('-RRB-', "PUNCT"), ("NFP", "PUNCT"),
    ("HYPH", "PUNCT")
    , ("FW", "X"), ("LS", "X"), ("XX", "X"), ("ADD", "X"), ("AFX", "X"), ("GW", "X")
])
UNIVERSAL_MAP = dict([
    ("ADP", "SCONJ/ADP"),
    ("SCONJ", "SCONJ/ADP")
])
# # nltk
# df_pos['pos_nltk'] = df_pos['sentence'].apply(lambda x: [pos[1] for pos in nltk.pos_tag(word_tokenize(x))])
# print(df_pos['pos_nltk'])
# df_pos['pos_nltk_univ'] = df_pos['pos_nltk'].apply(
#     lambda x: [PENN_TREEBANK_TO_UNIVERSAL_MAP[pos] if pos in PENN_TREEBANK_TO_UNIVERSAL_MAP else "[UNK]" for pos in
#                x])
# print(df_pos['pos_nltk_univ'])
# print(word_tokenize('I love python'))
# df_pos['nltk_index'] = df_pos['sentence'].apply(
#     lambda x: [x.index(tok) if tok in x else 'NA' for tok in word_tokenize(x)])
# print(df_pos['nltk_index'])
#
# # stanza
# nlp_stanza = stanza.Pipeline(lang='en', processors='tokenize,pos')
# df_pos['pos_stanza'] = df_pos['sentence'].apply(lambda x: list(
#     np.concatenate(np.array([[word.xpos for word in s.words] for s in nlp_stanza(x).sentences]), axis=0)))
# print(df_pos['pos_stanza'])
# df_pos['pos_stanza_univ'] = df_pos['pos_stanza'].apply(
#     lambda x: [PENN_TREEBANK_TO_UNIVERSAL_MAP[pos] if pos in PENN_TREEBANK_TO_UNIVERSAL_MAP else "[UNK]" for pos in
#                x])
# print(df_pos['pos_stanza_univ'])
# df_pos['stanza_index'] = df_pos['sentence'].apply(lambda x: [x.index(tok) if tok in x else 'NA' for tok in
#                                                              np.concatenate(np.array(
#                                                                  [[word.text for word in s.words] for s in
#                                                                   nlp_stanza(x).sentences]), axis=0)])
# print(df_pos['stanza_index'])


# spacy
nlp_spacy = en_core_web_sm.load()
nlp_spacy.tokenizer = RevisedTreeBankWordTokenizerVocab(nlp_spacy.vocab)
df_pos['pos_spacy'] = df_pos['sentence'].apply(lambda x: [word.pos_ for word in nlp_spacy(x)])
print(df_pos['pos_spacy'])
df_pos['pos_spacy_univ'] = df_pos['pos_spacy'].apply(lambda x:
                                                     [UNIVERSAL_MAP[pos] if pos in UNIVERSAL_MAP else pos for pos in
                                                      x])
print(df_pos['pos_spacy_univ'])
df_pos['spacy_index'] = df_pos['sentence'].apply(
    lambda x: [x.index(tok.text) if tok.text in x else 'NA' for tok in nlp_spacy(x)])

print(df_pos['spacy_index'])

# flair
tagger = SequenceTagger.load('pos')
df_pos['pos_flair'] = df_pos['sentence']
df_pos['flair_index'] = df_pos['sentence']
for i in range(len(df_pos)):
    if i % 100 == 0:
        print(i)
    sent_tokens = nltk.sent_tokenize(df_pos.loc[i, 'sentence'])
    sentences = [Sentence(i) for i in sent_tokens]
    tagger.predict(sentences)
    df_pos.loc[i, 'pos_flair'] = str(list(
        np.concatenate(np.array([[word.get_tag('pos').value for word in sentence] for sentence in sentences]), axis=0)))
    df_pos.loc[i, 'flair_index'] = str([df_pos.loc[i, 'sentence'].index(tok) if tok in df_pos.loc[i, 'sentence'] else 'NA' for tok in
        np.concatenate(np.array([[word.text for word in sentence] for sentence in sentences]), axis=0)])


print(df_pos['pos_flair'])
df_pos['pos_flair_univ'] = df_pos['pos_flair'].apply(lambda x:
                                                     [UNIVERSAL_MAP[pos] if pos in UNIVERSAL_MAP else pos for pos in
                                                      ast.literal_eval(x)])
print(df_pos['pos_flair'])


# textblob
df_pos['pos_textblob'] = df_pos['sentence'].apply(lambda x: [i[1] for i in TextBlob(x).tags])
print(df_pos['pos_textblob'])
df_pos['pos_textblob_univ'] = df_pos['pos_textblob'].apply(lambda x:
                                                           [PENN_TREEBANK_TO_UNIVERSAL_MAP[
                                                                pos] if pos in PENN_TREEBANK_TO_UNIVERSAL_MAP else pos
                                                            for pos in
                                                            x])

print(df_pos['pos_textblob_univ'])
df_pos['textblob_index'] = df_pos['sentence'].apply(
    lambda x: [x.index(tok.text) if tok in x else 'NA' for tok in [i[0] for i in TextBlob(x).tags]])

# gc

df_pos_gc = pd.read_csv('pos_results_gc.csv')

nb_len_before = 0
nb_len_after = 0
for i in range(len(df_pos)):
    if i % 100 == 0:
        print(i)
    nb_len_after += len(df_pos.loc[i, 'sentence'])
    labels = df_pos_gc[(df_pos_gc.offset <= nb_len_after) & (df_pos_gc.offset >= nb_len_before)]['pos'].tolist()
    nb_len_before += len(df_pos.loc[i, 'sentence'])
    df_pos.loc[i, 'pos_gc'] = str(labels)
    print(df_pos.loc[i, 'pos_gc'])

print(df_pos['pos_gc'])

df_pos['pos_gc_univ'] = df_pos['pos_gc'].apply(lambda x:
                                               [UNIVERSAL_MAP[pos] if pos in UNIVERSAL_MAP else pos for pos in
                                                ast.literal_eval(x)])
print()
print(df_pos['pos_gc_univ'])

df_pos.to_csv('sentences_to_GT_POS_corr_temp.csv')

# def different(spacy, nltk, stanza):
#     if 0 in [1 if spacy_val == nltk_val == stanza_val else 0 for spacy_val, nltk_val, stanza_val in
#              zip(spacy, nltk, stanza)]:
#         return 0
#     else:
#         return 1
#
#
# def different_tokens(sentence_tok, spacy, nltk, stanza):
#     return [sentence_tok_val if spacy_val == nltk_val == stanza_val else str("### " + sentence_tok_val) for
#             sentence_tok_val, spacy_val, nltk_val, stanza_val in zip(sentence_tok, spacy, nltk, stanza)]
#
#
# df_pos['is_different_tokens'] = df_pos[['pos_spacy_univ', 'pos_nltk_univ', 'pos_stanza_univ']].apply(
#     lambda x: different(x[0], x[1], x[2]), axis=1)
# df_pos['different_tokens'] = df_pos[['sentence_tok', 'pos_spacy_univ', 'pos_nltk_univ', 'pos_stanza_univ']].apply(
#     lambda x: different_tokens(x[0], x[1], x[2], x[3]), axis=1)
# print(df_pos['is_different_tokens'])
# print(df_pos['different_tokens'])
# df_pos['gt_help'] = df_pos[['different_tokens', 'GT']].apply(
#     lambda x: [(i, j) for i, j in zip(x[0], x[1])], axis=1)
# print(df_pos['gt_help'])
# df_pos.to_csv('sentences_to_GT_POS_corr_temp.csv')
