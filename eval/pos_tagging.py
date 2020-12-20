import pandas as pd
import nltk
from tokenizer.treebankwordtokenizer import RevisedTreeBankWordTokenizer
from tokenizer.treebankwordtokenizer_spacy_whitespace import RevisedTreeBankWordTokenizerVocab
import stanza
import en_core_web_sm
# stanza.download('en')
import spacy
from sklearn.metrics import precision_score, recall_score, accuracy_score

spacy.load('en_core_web_sm')
import numpy as np

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
    ("NNPS", "PROPN"), ("SYM", "SYM"), ("RP", "PART"),
    (".", "PUNCT"), ("UH", "INTJ"), ("POS", "PRON"), ("PRP$", "PRON"), ("WDT", "DET"),
    ("WP", "PRON"), ("TO", "SCONJ/ADP")
])
UNIVERSAL_MAP = dict([
    ("ADP", "SCONJ/ADP"),
    ("SCONJ", "SCONJ/ADP")
])

sentences = ['URL decoding in Javascript', 'How to register a JavaScript callback in a Java Applet ?',
             'I would not disagree with this .',
             'You can deploy both WARs in the same EAR and put common resources in the EAR . Then put the appropriate dependencies in the manifest of the web apps to link to the jar files in the ear .',
             'Which JavaScript library you recommend to use with Java EE + Struts + iBatis ?']

sentences_gt = [["^", "V", "P", "^"], ["R", "P", "V", "D", "^", "N", "P", "D", "^", "N", ","],
                ["O", "V", "R", "V", "P", "D", ","],
                ["O", "V", "P", "D", "^", "P", "D", "A", "^", "&", "V", "N", "N", "P", "D", "^", ",", "R", "V", "D",
                 "A", "N", "P", "D", "N", "P", "D", "N", "N", "P", "N", "P", "D", "^", "N", "P", "D", "^", ","],
                ["O", "^", "N", "O", "V", "P", "V", "P", "^", "^", "&", "^", "&", "^", ","]]
print('GT before mapping:' + str(sentences_gt))
sentences_gt_ptb = [[ARTICLE_TO_UNIVERSAL_MAP[pos] if pos in ARTICLE_TO_UNIVERSAL_MAP else "[UNK]" for pos in sent_pos]
                    for sent_pos
                    in sentences_gt]
print('GT:' + str(sentences_gt_ptb))

tokenizer = RevisedTreeBankWordTokenizer()

# nltk
sentences_tok = [tokenizer.tokenize(sent) for sent in sentences]
pos_nltk = [[pos[1] for pos in nltk.pos_tag(sent)] for sent in sentences_tok]
print('nltk before mapping:' + str(pos_nltk))
pos_nltk_univ = [
    [PENN_TREEBANK_TO_UNIVERSAL_MAP[pos] if pos in PENN_TREEBANK_TO_UNIVERSAL_MAP else "[UNK]" for pos in
     sent_pos] for
    sent_pos in pos_nltk]
print('nltk:' + str(pos_nltk_univ))

# stanza
nlp_stanza = stanza.Pipeline(lang='en', processors='tokenize,pos', tokenize_pretokenized=True)
sentences_stanza = nlp_stanza(sentences_tok)
pos_stanza = [[word.xpos for word in s.words] for s in sentences_stanza.sentences]
print('stanza before mapping:' + str(pos_stanza))
pos_stanza_univ = [
    [PENN_TREEBANK_TO_UNIVERSAL_MAP[pos] if pos in PENN_TREEBANK_TO_UNIVERSAL_MAP else "[UNK]" for pos in
     sent_pos] for
    sent_pos in pos_stanza]
print('stanza:' + str(pos_stanza_univ))

# spacy
nlp_spacy = en_core_web_sm.load()
nlp_spacy.tokenizer = RevisedTreeBankWordTokenizerVocab(nlp_spacy.vocab)
sentences_spacy = [nlp_spacy(sentence) for sentence in sentences]
pos_spacy = [[word.pos_ for word in sentence] for sentence in sentences_spacy]
print('spacy before mapping:' + str(pos_spacy))
pos_spacy_univ = [
    [UNIVERSAL_MAP[pos] if pos in UNIVERSAL_MAP else pos for pos in
     sent_pos] for
    sent_pos in pos_spacy]
print('spacy:' + str(pos_spacy_univ))

data = {'sentences': sentences,
        'nltk': pos_nltk_univ,
        'spacy': pos_spacy_univ,
        'stanza': pos_stanza_univ,
        'GT': sentences_gt_ptb,
        }


def good_predictions(pred, gt):
    return sum([1 for pred_val, gt_val in zip(pred, gt) if pred_val == gt_val])


def good_predictions_nouns(pred, gt):
    return sum(
        [1 for pred_val, gt_val in zip(pred, gt) if (pred_val == gt_val and gt_val == 'NOUN' or gt_val == 'PROPN')])


# test set
df = pd.DataFrame(data)

df['num_tokens'] = pd.Series(df['GT']).apply(len)
df['nltk_nb_good_predictions'] = df[['nltk', 'GT']].apply(lambda x: good_predictions(x[0], x[1]), axis=1)
df['spacy_nb_good_predictions'] = df[['spacy', 'GT']].apply(lambda x: good_predictions(x[0], x[1]), axis=1)
df['stanza_nb_good_predictions'] = df[['stanza', 'GT']].apply(lambda x: good_predictions(x[0], x[1]), axis=1)
df['nltk_nb_good_predictions_nouns'] = df[['nltk', 'GT']].apply(lambda x: good_predictions_nouns(x[0], x[1]), axis=1)
df['spacy_nb_good_predictions_nouns'] = df[['spacy', 'GT']].apply(lambda x: good_predictions_nouns(x[0], x[1]), axis=1)
df['stanza_nb_good_predictions_nouns'] = df[['stanza', 'GT']].apply(lambda x: good_predictions_nouns(x[0], x[1]),
                                                                    axis=1)
df['num_tokens'] = pd.Series(df['GT']).apply(len)

df.to_csv('test_set_pos_tagging.csv')


df_final = pd.DataFrame(np.array([[np.sum(df['nltk_nb_good_predictions']) / np.sum(df['num_tokens']),
                                   np.mean(df[['nltk', 'GT']].apply(lambda x: recall_score(
                                       [1 if (gt_val == 'NOUN' or gt_val == 'PROPN') else 0 for gt_val in x[1]],
                                       [1 if (gt_val == 'NOUN' or gt_val == 'PROPN') else 0 for gt_val in x[0]]),
                                                                    axis=1)),
                                   np.mean(df[['nltk', 'GT']].apply(lambda x: precision_score(
                                       [1 if (gt_val == 'NOUN' or gt_val == 'PROPN') else 0 for gt_val in x[1]],
                                       [1 if (gt_val == 'NOUN' or gt_val == 'PROPN') else 0 for gt_val in x[0]]),
                                                                    axis=1)),
                                   np.mean(df[['nltk', 'GT']].apply(lambda x: accuracy_score(
                                       [1 if (gt_val == 'NOUN' or gt_val == 'PROPN') else 0 for gt_val in x[1]],
                                       [1 if (gt_val == 'NOUN' or gt_val == 'PROPN') else 0 for gt_val in x[0]]),
                                                                    axis=1))],
                                  [np.sum(df['spacy_nb_good_predictions']) / np.sum(df['num_tokens']),
                                   np.mean(df[['spacy', 'GT']].apply(lambda x: recall_score(
                                       [1 if (gt_val == 'NOUN' or gt_val == 'PROPN') else 0 for gt_val in x[1]],
                                       [1 if (gt_val == 'NOUN' or gt_val == 'PROPN') else 0 for gt_val in x[0]]),
                                                                     axis=1)),
                                   np.mean(df[['spacy', 'GT']].apply(lambda x: precision_score(
                                       [1 if (gt_val == 'NOUN' or gt_val == 'PROPN') else 0 for gt_val in x[1]],
                                       [1 if (gt_val == 'NOUN' or gt_val == 'PROPN') else 0 for gt_val in x[0]]),
                                                                     axis=1)),
                                   np.mean(df[['spacy', 'GT']].apply(lambda x: accuracy_score(
                                       [1 if (gt_val == 'NOUN' or gt_val == 'PROPN') else 0 for gt_val in x[1]],
                                       [1 if (gt_val == 'NOUN' or gt_val == 'PROPN') else 0 for gt_val in x[0]]),
                                                                     axis=1))],
                                  [np.sum(df['stanza_nb_good_predictions']) / np.sum(df['num_tokens']),
                                   np.mean(df[['stanza', 'GT']].apply(lambda x: recall_score(
                                       [1 if (gt_val == 'NOUN' or gt_val == 'PROPN') else 0 for gt_val in x[1]],
                                       [1 if (gt_val == 'NOUN' or gt_val == 'PROPN') else 0 for gt_val in x[0]]),
                                                                      axis=1)),
                                   np.mean(df[['stanza', 'GT']].apply(lambda x: precision_score(
                                       [1 if (gt_val == 'NOUN' or gt_val == 'PROPN') else 0 for gt_val in x[1]],
                                       [1 if (gt_val == 'NOUN' or gt_val == 'PROPN') else 0 for gt_val in x[0]]),
                                                                      axis=1)),
                                   np.mean(df[['stanza', 'GT']].apply(lambda x: accuracy_score(
                                       [1 if (gt_val == 'NOUN' or gt_val == 'PROPN') else 0 for gt_val in x[1]],
                                       [1 if (gt_val == 'NOUN' or gt_val == 'PROPN') else 0 for gt_val in x[0]]),
                                                                      axis=1))]]),
                        columns=['all tokens', 'nouns_recall', 'nouns_precision', 'nouns_accuracy'],
                        index=['nltk', 'spacy', 'stanza'])

df_final.to_csv('final.csv')
