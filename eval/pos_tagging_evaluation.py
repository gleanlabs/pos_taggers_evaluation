import spacy
from sklearn.metrics import precision_score, recall_score, accuracy_score
from source.tag_pos import _read_tag_map, map_results_to_universal_tags, _pos_tag_sentence
from source.pos_taggers_functions import split_labels_articles_that_need_to
from source.tokenizer_functions import tokenize
import nltk

spacy.load('en_core_web_sm')
import pandas as pd
import numpy as np
import ast

# df_Jo  = pd.read_csv('reviewing_dataset_Johanna.csv')
# df_Basel  = pd.read_csv('sentences_to_GT_POS_corrected_Basel.csv')
# for i in range(len(df_Jo)):
#     if i <=499:
#         df_Basel.loc[i, 'GT_POS'] = str([i[1] if type(i[1])==str else i[1][1] for i in ast.literal_eval(df_Jo.loc[i,'new_tagging_corrected'])])
#
# df_Basel.to_csv('sentences_to_GT_POS_corrected_Basel_Jo.csv')


df = pd.read_csv('sentences_to_GT_POS_corrected_Basel.csv')
mapping = _read_tag_map()
dict_mapping = mapping['ARTICLE-UNIV']
df['GT'] = df[['sentence', 'GT_POS']].apply(
    lambda x: split_labels_articles_that_need_to(
        [(i, j) for i, j in
         zip([item for sublist in [text.split(' ') for text in nltk.sent_tokenize(x[0])] for item in sublist],
             ast.literal_eval(x[1]))]), axis=1)
print('GT')
print(df['GT'])

# nltk
df['nltk'] = df['sentence'].apply(
    lambda x: [i[1] for i in map_results_to_universal_tags(_pos_tag_sentence('nltk', x), 'nltk')])
print('nltk')

df['same'] = df[['GT', 'nltk']].apply(lambda x: 1 if len(x[0]) == len(x[1]) else 0, axis=1)
df = df[df.same == 1]
print(len([item for sublist in df['nltk'].tolist() for item in sublist]))
print(len([item for sublist in df['GT'].tolist() for item in sublist]))

# stanza
df['stanza'] = df['sentence'].apply(
    lambda x: [i[1] for i in map_results_to_universal_tags(_pos_tag_sentence('stanza', x), 'stanza')])
print('stanza')
print(len([item for sublist in df['stanza'].tolist() for item in sublist]))
df['same'] = df[['GT', 'stanza']].apply(lambda x: 1 if len(x[0]) == len(x[1]) else 0, axis=1)
df = df[df.same == 1]

# spacy
df['spacy'] = df['sentence'].apply(
    lambda x: [i[1] for i in map_results_to_universal_tags(_pos_tag_sentence('spacy', x), 'spacy')])
print('spacy')

df['same'] = df[['GT', 'spacy']].apply(lambda x: 1 if len(x[0]) == len(x[1]) else 0, axis=1)
df = df[df.same == 1]

print(len([item for sublist in df['spacy'].tolist() for item in sublist]))


def good_predictions(pred, gt):
    return sum([1 for pred_val, gt_val in zip(pred, gt) if pred_val == gt_val])


def good_predictions_nouns(pred, gt):
    return sum(
        [1 for pred_val, gt_val in zip(pred, gt) if (pred_val == gt_val and gt_val == 'NOUN' or gt_val == 'PROPN')])


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
#
# df_final = pd.DataFrame(np.array([[np.sum(df['nltk_nb_good_predictions']) / np.sum(df['num_tokens']),
#                                    np.mean(df[['nltk', 'GT']].apply(lambda x: recall_score(
#                                        [1 if (gt_val == 'NOUN' or gt_val == 'PROPN') else 0 for gt_val in x[1]],
#                                        [1 if (gt_val == 'NOUN' or gt_val == 'PROPN') else 0 for gt_val in x[0]]),
#                                                                     axis=1)),
#                                    np.mean(df[['nltk', 'GT']].apply(lambda x: precision_score(
#                                        [1 if (gt_val == 'NOUN' or gt_val == 'PROPN') else 0 for gt_val in x[1]],
#                                        [1 if (gt_val == 'NOUN' or gt_val == 'PROPN') else 0 for gt_val in x[0]]),
#                                                                     axis=1)),
#                                    np.mean(df[['nltk', 'GT']].apply(lambda x: accuracy_score(
#                                        [1 if (gt_val == 'NOUN' or gt_val == 'PROPN') else 0 for gt_val in x[1]],
#                                        [1 if (gt_val == 'NOUN' or gt_val == 'PROPN') else 0 for gt_val in x[0]]),
#                                                                     axis=1))],
#                                   [np.sum(df['spacy_nb_good_predictions']) / np.sum(df['num_tokens']),
#                                    np.mean(df[['spacy', 'GT']].apply(lambda x: recall_score(
#                                        [1 if (gt_val == 'NOUN' or gt_val == 'PROPN') else 0 for gt_val in x[1]],
#                                        [1 if (gt_val == 'NOUN' or gt_val == 'PROPN') else 0 for gt_val in x[0]]),
#                                                                      axis=1)),
#                                    np.mean(df[['spacy', 'GT']].apply(lambda x: precision_score(
#                                        [1 if (gt_val == 'NOUN' or gt_val == 'PROPN') else 0 for gt_val in x[1]],
#                                        [1 if (gt_val == 'NOUN' or gt_val == 'PROPN') else 0 for gt_val in x[0]]),
#                                                                      axis=1)),
#                                    np.mean(df[['spacy', 'GT']].apply(lambda x: accuracy_score(
#                                        [1 if (gt_val == 'NOUN' or gt_val == 'PROPN') else 0 for gt_val in x[1]],
#                                        [1 if (gt_val == 'NOUN' or gt_val == 'PROPN') else 0 for gt_val in x[0]]),
#                                                                      axis=1))],
#                                   [np.sum(df['stanza_nb_good_predictions']) / np.sum(df['num_tokens']),
#                                    np.mean(df[['stanza', 'GT']].apply(lambda x: recall_score(
#                                        [1 if (gt_val == 'NOUN' or gt_val == 'PROPN') else 0 for gt_val in x[1]],
#                                        [1 if (gt_val == 'NOUN' or gt_val == 'PROPN') else 0 for gt_val in x[0]]),
#                                                                       axis=1)),
#                                    np.mean(df[['stanza', 'GT']].apply(lambda x: precision_score(
#                                        [1 if (gt_val == 'NOUN' or gt_val == 'PROPN') else 0 for gt_val in x[1]],
#                                        [1 if (gt_val == 'NOUN' or gt_val == 'PROPN') else 0 for gt_val in x[0]]),
#                                                                       axis=1)),
#                                    np.mean(df[['stanza', 'GT']].apply(lambda x: accuracy_score(
#                                        [1 if (gt_val == 'NOUN' or gt_val == 'PROPN') else 0 for gt_val in x[1]],
#                                        [1 if (gt_val == 'NOUN' or gt_val == 'PROPN') else 0 for gt_val in x[0]]),
#                                                                       axis=1))]]),
#                         columns=['all tokens', 'nouns_recall', 'nouns_precision', 'nouns_accuracy'],
#                         index=['nltk', 'spacy', 'stanza'])
#
# df_final.to_csv('final.csv')

flat_list_nltk = [item for sublist in df['nltk'].tolist() for item in sublist]
flat_list_spacy = [item for sublist in df['spacy'].tolist() for item in sublist]
flat_list_stanza = [item for sublist in df['stanza'].tolist() for item in sublist]
flat_list_gt = [item for sublist in df['GT'].tolist() for item in sublist]

from sklearn.metrics import confusion_matrix

array_nltk = confusion_matrix(flat_list_gt, flat_list_nltk, labels=list(set(flat_list_nltk)))
array_spacy = confusion_matrix(flat_list_gt, flat_list_spacy, labels=list(set(flat_list_spacy)))
array_stanza = confusion_matrix(flat_list_gt, flat_list_stanza, labels=list(set(flat_list_stanza)))

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

df_cm = pd.DataFrame(array_nltk, index=[i for i in list(set(flat_list_nltk))],
                     columns=[i for i in list(set(flat_list_nltk))])
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True)
plt.title('nltk confusion matrix')
plt.show()

df_cm = pd.DataFrame(array_spacy, index=[i for i in list(set(flat_list_spacy))],
                     columns=[i for i in list(set(flat_list_spacy))])
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True)
plt.title('spacy confusion matrix')
plt.show()

df_cm = pd.DataFrame(array_stanza, index=[i for i in list(set(flat_list_stanza))],
                     columns=[i for i in list(set(flat_list_stanza))])
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True)
plt.title('stanza confusion matrix')
plt.show()
