import pandas as pd
import nltk
from tokenizer.treebankwordtokenizer import RevisedTreeBankWordTokenizer
from tokenizer.treebankwordtokenizer_spacy import RevisedTreeBankWordTokenizerVocab
import stanza
import en_core_web_sm
# stanza.download('en')
import spacy
from sklearn.metrics import precision_score, recall_score, accuracy_score
import ast

spacy.load('en_core_web_sm')
import numpy as np

df_pos = pd.read_csv('sentences_to_GT_POS.csv')
df_pos['sentence_tok'] = df_pos['sentence'].apply(lambda x: x.split())
df_pos['GT'] = df_pos['tagged_tokens_GT'].apply(lambda x: [i[1] for i in ast.literal_eval(x)])

df_pos['GT'] = df_pos['tagged_tokens_GT'].apply(lambda x: [i[1] for i in ast.literal_eval(x)])

# for added_so_far, sentence, tokens, tokens_pos in rows:
#     token_to_tag_list = list(zip(tokens, tokens_pos))
#     output_columns.append((added_so_far, "(=SENTENCE     =)", sentence))
#     output_columns.append((added_so_far, "(=ORIGINAL  POS=)", token_to_tag_list))
#     output_columns.append((added_so_far, "(=CORRECTED POS=)", token_to_tag_list))
#     output_columns.append(("", "", ""))

ARTICLE_TO_UNIVERSAL_MAP = dict([
    ("&", "CONJ"), ("$", "NUM"), ("D", "DET"), ("P", "SCONJ"), ("A", "ADJ"), ("N", "NOUN"),
    ("O", "PRON"), ("R", "ADV"), ("V", "VERB"), ("^", "PROPN"), ("G", "SYM"), ("!", "INTJ"), ("T", "PART"),
    ("X", "DET"), (",", "PUNCT")
])

PENN_TREEBANK_TO_UNIVERSAL_MAP = dict([
    ("CC", "CCONJ"), ("CD", "NUM"), ("DT", "DET"), ("PDT", "DET"), ("FW", "X"), ("IN", "SCONJ"), ("JJ", "ADJ"),
    ("JJR", "ADJ"),
    ("JJS", "ADJ"), ("NN", "NOUN"), ("NNS", "NOUN"), ("MD", "VERB"),
    ("PRT", "PRT"), ("PRP", "PRON"), ("RB", "ADV"), ("RBR", "ADV"), ("RBS", "ADV"), ("WRB", "ADV"), ("VB", "VERB"),
    ("VBD", "VERB"),
    ("VBG", "VERB"), ("VBN", "VERB"), ("VBP", "VERB"), ("VBZ", "VERB"), ("NNP", "PROPN"),
    ("NNPS", "PROPN"), ("SYM", "SYM"), ("RP", "PART"),
    (".", "PUNCT"), ("UH", "INTJ"), ("POS", "PRON"), ("PRP$", "PRON"), ("WDT", "DET"),
    ("WP", "PRON"), ("TO", "SCONJ")
])
UNIVERSAL_MAP = dict([
    ("ADP", "SCONJ")
])

# nltk
df_pos['pos_nltk'] = df_pos['sentence_tok'].apply(lambda x: [pos[1] for pos in nltk.pos_tag(x)])
print(df_pos['pos_nltk'])
df_pos['pos_nltk_univ'] = df_pos['pos_nltk'].apply(
    lambda x: [PENN_TREEBANK_TO_UNIVERSAL_MAP[pos] if pos in PENN_TREEBANK_TO_UNIVERSAL_MAP else "[UNK]" for pos in
               x])
print(df_pos['pos_nltk_univ'])

# stanza
nlp_stanza = stanza.Pipeline(lang='en', processors='tokenize,pos', tokenize_pretokenized=True)
df_pos['pos_stanza'] = df_pos['sentence_tok'].apply(lambda x: list(
    np.concatenate(np.array([[word.xpos for word in s.words] for s in nlp_stanza([x]).sentences]), axis=0)))
print(df_pos['pos_stanza'])
df_pos['pos_stanza_univ'] = df_pos['pos_stanza'].apply(
    lambda x: [PENN_TREEBANK_TO_UNIVERSAL_MAP[pos] if pos in PENN_TREEBANK_TO_UNIVERSAL_MAP else "[UNK]" for pos in
               x])
print(df_pos['pos_stanza_univ'])

# spacy
nlp_spacy = en_core_web_sm.load()
nlp_spacy.tokenizer = RevisedTreeBankWordTokenizerVocab(nlp_spacy.vocab)
df_pos['pos_spacy'] = df_pos['sentence_tok'].apply(lambda x: [word.pos_ for word in nlp_spacy(x)])
print(df_pos['pos_spacy'])
df_pos['pos_spacy_univ'] = df_pos['pos_spacy'].apply(lambda x:
                                                     [UNIVERSAL_MAP[pos] if pos in UNIVERSAL_MAP else pos for pos in
                                                      x])
print(df_pos['pos_spacy_univ'])


def different(spacy, nltk, stanza):
    if 0 in [1 if spacy_val == nltk_val == stanza_val else 0 for spacy_val, nltk_val, stanza_val in
             zip(spacy, nltk, stanza)]:
        return 0
    else:
        return 1


def different_tokens(sentence_tok, spacy, nltk, stanza):
    return [sentence_tok_val if spacy_val == nltk_val == stanza_val else str("### " + sentence_tok_val) for
            sentence_tok_val, spacy_val, nltk_val, stanza_val in zip(sentence_tok, spacy, nltk, stanza)]


df_pos['is_different_tokens'] = df_pos[['pos_spacy_univ', 'pos_nltk_univ', 'pos_stanza_univ']].apply(
    lambda x: different(x[0], x[1], x[2]), axis=1)
df_pos['different_tokens'] = df_pos[['sentence_tok', 'pos_spacy_univ', 'pos_nltk_univ', 'pos_stanza_univ']].apply(
    lambda x: different_tokens(x[0], x[1], x[2], x[3]), axis=1)
print(df_pos['is_different_tokens'])
print(df_pos['different_tokens'])
df_pos['gt_help'] = df_pos[['different_tokens', 'GT']].apply(
    lambda x: [(i, j) for i, j in zip(x[0], x[1])], axis=1)
print(df_pos['gt_help'])
df_pos.to_csv('sentences_to_GT_POS_corr_temp.csv')

# good format

list_of_lists = []
for i in range(len(df_pos)):
    list_of_lists.append([df_pos.loc[i, 'row'], "(=SENTENCE     =)", df_pos.loc[i, 'sentence']])
    list_of_lists.append([df_pos.loc[i, 'row'], "(=ORIGINAL  POS=)", df_pos.loc[i, 'gt_help']])
    list_of_lists.append([df_pos.loc[i, 'row'], "(=CORRECTED POS=)", df_pos.loc[i, 'gt_help']])
    list_of_lists.append([df_pos.loc[i, 'row'], '', ''])

df_corrected = pd.DataFrame(list_of_lists)
df_corrected.to_csv('sentences_to_GT_POS_corr.csv')
