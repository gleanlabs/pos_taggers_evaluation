import spacy
from source.tag_pos import _read_tag_map, map_results_to_universal_tags, _pos_tag_sentence
from source.pos_taggers_functions import split_labels_articles_that_need_to, _split_composite_pos_tokens
import nltk

spacy.load('en_core_web_sm')
import pandas as pd
import numpy as np
import ast
from sklearn.metrics import classification_report

# df_Jo  = pd.read_csv('reviewing_dataset_Johanna.csv')
# df_Basel  = pd.read_csv('sentences_to_GT_POS_corrected_Basel.csv')
# for i in range(len(df_Jo)):
#     if i <=499:
#         df_Basel.loc[i, 'GT_POS'] = str([i[1] if type(i[1])==str else i[1][1] for i in ast.literal_eval(df_Jo.loc[i,'new_tagging'])])
#
# df_Basel.to_csv('sentences_to_GT_POS_corrected_Basel_Jo.csv')


df = pd.read_csv('sentences_to_GT_POS_corrected_Basel.csv')
mapping = _read_tag_map()
dict_mapping = mapping['ARTICLE-UNIV']
df['GT'] = df[['sentence', 'GT_POS']].apply(
    lambda x: [dict_mapping[gt] for gt in split_labels_articles_that_need_to(
        [(i, j) for i, j in
         zip([item for sublist in [text.split(' ') for text in nltk.sent_tokenize(x[0])] for item in sublist],
             ast.literal_eval(x[1]))])], axis=1)
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

df.to_csv('test_set_pos_tagging.csv')


df = pd.read_csv('test_set_pos_tagging.csv')[500:]

df['nltk'] = df['nltk'].apply(lambda x: ast.literal_eval(x))
df['spacy'] = df['spacy'].apply(lambda x: ast.literal_eval(x))
df['stanza'] = df['stanza'].apply(lambda x: ast.literal_eval(x))
df['GT'] = df['GT'].apply(lambda x: ast.literal_eval(x))

df['agree'] = df[['nltk', 'spacy', 'stanza']].apply(
    lambda x: [1 if nl == sp == st else 0 for nl, sp, st in zip(x[0], x[1], x[2])], axis=1)
flat_list_nltk = [item for sublist in
                  df[['nltk', 'agree']].apply(lambda x: [nltk for nltk, agree in zip(x[0], x[1]) if agree == 0],
                                              axis=1).tolist() for
                  item in sublist]
flat_list_spacy = [item for sublist in
                   df[['spacy', 'agree']].apply(lambda x: [nltk for nltk, agree in zip(x[0], x[1]) if agree == 0],
                                                axis=1).tolist() for
                   item in sublist]
flat_list_stanza = [item for sublist in
                    df[['stanza', 'agree']].apply(lambda x: [nltk for nltk, agree in zip(x[0], x[1]) if agree == 0],
                                                  axis=1).tolist() for
                    item in sublist]
flat_list_gt = [item for sublist in
                df[['GT', 'agree']].apply(lambda x: [nltk for nltk, agree in zip(x[0], x[1]) if agree == 0],
                                          axis=1).tolist() for item in
                sublist]
print(len(flat_list_nltk) == len(flat_list_spacy) == len(flat_list_stanza) == len(flat_list_gt))
from sklearn.metrics import confusion_matrix

array_nltk = confusion_matrix(flat_list_gt, flat_list_nltk, labels=list(set(flat_list_gt)))
array_spacy = confusion_matrix(flat_list_gt, flat_list_spacy, labels=list(set(flat_list_gt)))
array_stanza = confusion_matrix(flat_list_gt, flat_list_stanza, labels=list(set(flat_list_gt)))

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

df_cm = pd.DataFrame(array_nltk, index=[i for i in list(set(flat_list_gt))],
                     columns=[i for i in list(set(flat_list_gt))])
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, fmt='g')
plt.title('nltk confusion matrix_'+str(np.sum(array_nltk)))
plt.xlabel("predicted")
plt.ylabel("actual")
plt.show()
print("NLTK:")
print(classification_report(flat_list_gt, flat_list_nltk,  labels=list(set(flat_list_gt))))




df_cm = pd.DataFrame(array_spacy, index=[i for i in list(set(flat_list_gt))],
                     columns=[i for i in list(set(flat_list_gt))])
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, fmt='g')
plt.title('spacy confusion matrix_'+str(np.sum(array_spacy)))
plt.xlabel("predicted")
plt.ylabel("actual")
plt.show()
print("SPACY:")
print(classification_report(flat_list_gt, flat_list_spacy,  labels=list(set(flat_list_gt))))


df_cm = pd.DataFrame(array_stanza, index=[i for i in list(set(flat_list_gt))],
                     columns=[i for i in list(set(flat_list_gt))])
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, fmt='g')
plt.title('stanza confusion matrix_'+str(np.sum(array_stanza)))
plt.xlabel("predicted")
plt.ylabel("actual")
plt.show()
print("STANZA:")
print(classification_report(flat_list_gt, flat_list_stanza,  labels=list(set(flat_list_gt))))





# why PUNCT/SYM being predicted PROPN

# df = pd.read_csv('test_set_pos_tagging.csv')[500:]
#
# df['nltk'] = df['nltk'].apply(lambda x: ast.literal_eval(x))
# df['spacy'] = df['spacy'].apply(lambda x: ast.literal_eval(x))
# df['stanza'] = df['stanza'].apply(lambda x: ast.literal_eval(x))
# df['GT'] = df['GT'].apply(lambda x: ast.literal_eval(x))
#
# df['agree'] = df[['nltk', 'GT']].apply(
#     lambda x: [1 if (nl == 'PROPN' and gt == 'PUNCT/SYM') else 0 for nl, gt in zip(x[0], x[1])], axis=1)
#
# df['gt_tok'] = df['sentence'].apply(
#     lambda x: [item for sublist in [_split_composite_pos_tokens(text.split(' ')) for text in nltk.sent_tokenize(x)] for
#                item in sublist])
#
# df['sent_tagged'] = df[['gt_tok', 'GT']].apply(lambda x: [(val1, val2) for val1, val2 in zip(x[0], x[1])], axis=1)
#
# df['sent_tagged_nltk'] = df[['gt_tok', 'nltk']].apply(lambda x: [(val1, val2) for val1, val2 in zip(x[0], x[1])], axis=1)
#
# df['tok'] = df[
#     ['gt_tok', 'nltk', 'agree']].apply(lambda x: [tok for tok, nltk, agree in zip(x[0], x[1], x[2]) if agree == 1],
#                                        axis=1)
#
# df.to_csv('checks.csv')

# print(nltk.pos_tag("Detecting EOF / TCP teardown using Java sockets from Javascript".split(' ')))
# # print(nltk.pos_tag("I need some ideas on how I can best solve this problem . I have a JBoss Seam application running on JBoss 4.3 . 3 What a small portion of this application does is generate an html and a pdf document based on an Open Office template . The files_Basel that are generated I put inside / tmp on the filesystem . I have tried with System.getProperties ( +' tmp.dir ') and some other options , and they always return $JBOSS_HOME / bin I would like to choose the path $JBOSS_HOME / $DEPLOY / myEAR.ear / myWAR.war / WhateverLocationHere However , I don't know how I can programatically choose path without giving an absolute path , or setting $JBOSS_HOME and $DEPLOY . Anybody know how I can do this ? The second question ; I want to easily preview these generated files_Basel . Either through JavaScript , or whatever is the easiest way . However , JavaScript cannot access the filesystem on the server , so I cannot open the file through JavaScript . Any easy solutions out there ?".split(' ')))
# print(nltk.pos_tag("Thank you !!! it was driving me crazy !!! :) About the manual ... mhhhh ... mmhh well .. Beside having to write my own product user manual ( which I hate and usually I avoid ) do I have to read others them too ??? :) :) :) ;) Thanks again".split(' ')))
