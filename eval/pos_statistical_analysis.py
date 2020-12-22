import pandas as pd
from collections import Counter
import seaborn as sns
import numpy as np
import ast
import matplotlib.pyplot as plt
from sklearn import preprocessing

# df_pos = pd.read_csv('sentences_to_GT_POS_corr_temp.csv')
# df_pos = df_pos[['row', 'sentence', 'GT', 'sentence_tok', 'GT_index', 'pos_nltk_univ', 'nltk_index', 'pos_stanza_univ',
#                  'stanza_index', 'pos_spacy_univ', 'spacy_index', 'pos_flair_univ', 'flair_index', 'pos_textblob_univ',
#                  'textblob_index', 'pos_gc_univ', 'gc_index']]
# df_pos['gt_new'] = df_pos['sentence']
# df_pos['libraries_pred'] = df_pos['sentence']
#
#
# def most_frequent(List):
#     occurence_count = Counter(List)
#     return occurence_count.most_common(1)[0][0]
#
#
# for i in range(len(df_pos)):
#     list_tokens = []
#     list_libraries_predictions = []
#     for tok, gt, index in zip(
#             ast.literal_eval(df_pos.loc[i, 'sentence_tok']),
#             ast.literal_eval(df_pos.loc[i, 'GT']),
#             ast.literal_eval(df_pos.loc[i, 'GT_index'])):
#         gc_index = ast.literal_eval(df_pos.loc[i, 'gc_index'])
#         nltk_index = ast.literal_eval(df_pos.loc[i, 'nltk_index'])
#         stanza_index = ast.literal_eval(df_pos.loc[i, 'stanza_index'])
#         spacy_index = ast.literal_eval(df_pos.loc[i, 'spacy_index'])
#         textblob_index = ast.literal_eval(df_pos.loc[i, 'textblob_index'])
#         if (index in spacy_index) & (
#                 index in stanza_index) & (
#                 index in gc_index) & (
#                 index in nltk_index) & (
#                 index in textblob_index):
#             gc = ast.literal_eval(df_pos.loc[i, 'pos_gc_univ'])
#             nltk = ast.literal_eval(df_pos.loc[i, 'pos_nltk_univ'])
#             stanza = ast.literal_eval(df_pos.loc[i, 'pos_stanza_univ'])
#             spacy = ast.literal_eval(df_pos.loc[i, 'pos_spacy_univ'])
#             textblob = ast.literal_eval(df_pos.loc[i, 'pos_textblob_univ'])
#             preds = [gt, gc[gc_index.index(index)], nltk[nltk_index.index(index)], stanza[stanza_index.index(index)],
#                      spacy[spacy_index.index(index)], textblob[textblob_index.index(index)]]
#             most_frequent_val = most_frequent(preds)
#             list_tokens.append(
#                 [tok, gt, most_frequent_val, preds.count(most_frequent_val),
#                  len(list(set(preds))),
#                  most_frequent_val == gt,
#                  preds])
#     df_pos.loc[i, 'gt_new'] = str(list_tokens)
#
# print(df_pos['gt_new'])
# df_pos.to_csv('sentences_to_GT_POS_corr_stats.csv')

# df_pos = pd.read_csv('sentences_to_GT_POS_corr_stats.csv')
# df_pos['gt_new'] = df_pos['gt_new'].apply(lambda x: ast.literal_eval(x))
# df_pos_exp = df_pos.explode('gt_new')
# df_pos_exp['is_gt_new_list'] = df_pos_exp['gt_new'].apply(lambda x: type(x))
# df_pos_exp = df_pos_exp[df_pos_exp.is_gt_new_list == list]
# df_pos_exp[['tok', 'gt', 'most_frequent', 'count_most_frequent', 'count_uniques', 'is_most_frequent_gt',
#             'libraries_pred']] = pd.DataFrame(df_pos_exp.gt_new.tolist(), index=df_pos_exp.index)
# df_pos_exp.to_csv('sentences_to_GT_POS_corr_stats_new_gt.csv')
#
#
#
# df_pos = pd.read_csv('sentences_to_GT_POS_corr_stats_new_gt.csv')
# df_pos['libraries_pred'] = df_pos['libraries_pred'].apply(lambda x: ast.literal_eval(x))
# df_pos = df_pos[['row', 'libraries_pred', 'gt_new']]
# df_pos['is_gt_new_list'] = df_pos['libraries_pred'].apply(lambda x: type(x))
# df_pos = df_pos[df_pos.is_gt_new_list == list]
# df_pos[['GT', 'GC', 'nltk', 'stanza', 'spacy', 'textblob']] = pd.DataFrame(df_pos.libraries_pred.tolist(), index=df_pos.index)
# df_pos.to_csv('sentences_to_GT_POS_corr_stats_libraries.csv')


# PLOTS
# df_pos_exp = pd.read_csv('sentences_to_GT_POS_corr_stats_new_gt.csv')
# f, ax = plt.subplots(figsize=(16, 12))
# sns.kdeplot(data=df_pos_exp, x="count_most_frequent", hue="is_most_frequent_gt")
# plt.title('When a major vote equals the GT or not, distributions of counting votes')
# plt.show()
#
# # with uniques
# g = sns.FacetGrid(data=df_pos_exp, col="is_most_frequent_gt", hue='count_uniques', height=5, aspect=0.5)
# g = g.map(sns.kdeplot, 'count_most_frequent', shade=True)
# g.add_legend()
# plt.show()
#
# f, ax = plt.subplots(figsize=(13, 10))
# sns.kdeplot(data=df_pos_exp, x="count_most_frequent", hue="most_frequent")
# plt.title('How libraries tend to agree for each tag,  distributions of counting votes')
# plt.show()
#
# f, ax = plt.subplots(figsize=(13, 10))
# sns.kdeplot(data=df_pos_exp[df_pos_exp.is_most_frequent_gt==1], x="count_most_frequent", hue="gt")
# plt.title('How libraries tend to agree with the GT, distribution of counting votes')
# plt.show()
#
# df_pos_exp_filt = df_pos_exp[df_pos_exp['most_frequent'].isin(['NOUN', 'PROPN', 'ADJ'])]
# f, ax = plt.subplots(figsize=(13, 10))
# sns.kdeplot(data=df_pos_exp_filt[df_pos_exp_filt.is_most_frequent_gt==1], x="count_most_frequent", hue="most_frequent")
# plt.title('How libraries tend to agree with the GT, distribution of counting votes (filt)')
# plt.show()
#
#
# import matplotlib.pyplot as plt
#
# # count plot on single categorical variable
# sns.countplot(x='is_most_frequent_gt', data=df_pos_exp)
#
# # Show the plot
# plt.show()
#
#
# f, ax = plt.subplots(figsize=(16, 12))
# df_dummies = pd.get_dummies(df_pos_exp[['most_frequent', 'gt']])
# print(df_dummies)
#
# x = df_dummies.values
# correlation_matrix = np.corrcoef(x.T)
# print(correlation_matrix)
#
# # plot the heatmap
# sns.heatmap(correlation_matrix,
#             xticklabels=df_dummies.columns,
#             yticklabels=df_dummies.columns)
# plt.show()
#
#
# f, ax = plt.subplots(figsize=(16, 12))
# df_pos_exp_filt2 = df_pos_exp[~(df_pos_exp['most_frequent'] == df_pos_exp['gt'])]
# df_dummies = pd.get_dummies(df_pos_exp_filt2[['most_frequent', 'gt']])
#
# x = df_dummies.values
# correlation_matrix = np.corrcoef(x.T)
#
# # plot the heatmap
# sns.heatmap(correlation_matrix,
#             xticklabels=df_dummies.columns,
#             yticklabels=df_dummies.columns)
# plt.show()
#
# print(len(df_pos_exp))
# to_check = df_pos_exp[((df_pos_exp.count_most_frequent == 4) & (df_pos_exp.count_uniques <= 2)) | (
#             (df_pos_exp.count_most_frequent == 3) & (df_pos_exp.count_uniques <= 3)) | (
#                                   (df_pos_exp.count_most_frequent == 2) & (df_pos_exp.count_uniques <= 3))]
# print(len(to_check))


# PLOTS LIBRARIES
# df_pos_exp =  pd.read_csv('sentences_to_GT_POS_corr_stats_libraries.csv')
# print(df_pos_exp[['GT', 'GC', 'nltk', 'stanza', 'spacy', 'textblob']])
# stacked = df_pos_exp[['GT', 'GC', 'nltk', 'stanza', 'spacy', 'textblob']].stack().astype('category')
# df_pos_exp_encoded = stacked.cat.codes.unstack()
# print(df_pos_exp_encoded)
#
# f, ax = plt.subplots(figsize=(16, 12))
# x = df_pos_exp_encoded.values
# print(x)
# correlation_matrix = np.corrcoef(x.T)
# print(correlation_matrix)
# # plot the heatmap
# sns.heatmap(correlation_matrix,
#             xticklabels=df_pos_exp_encoded.columns,
#             yticklabels=df_pos_exp_encoded.columns, annot=True)
# plt.show()
#
#
# def disagree(libraries, pred):
#     labels = ['GT', 'GC', 'nltk', 'stanza', 'spacy', 'textblob']
#     libraries_dis = []
#     lib_labels = [(lab, val) for lab, val in zip(labels, libraries)]
#     for val in lib_labels:
#         if val[1] != pred:
#             libraries_dis.append(val[0])
#     return libraries_dis
#
# def disagree_together(libraries, pred):
#     labels = ['GT', 'GC', 'nltk', 'stanza', 'spacy', 'textblob']
#     libraries_dis = []
#     lib_labels = [(lab, val) for lab, val in zip(labels, libraries)]
#     libraries_dis_tog = []
#     for val in lib_labels:
#         if val[1] != pred:
#             libraries_dis_tog.append(val[0])
#     libraries_dis.append(' '.join(libraries_dis_tog))
#     return libraries_dis
#
#
# df_pos = pd.read_csv('sentences_to_GT_POS_corr_stats_new_gt.csv')
# df_pos['libraries_pred'] = df_pos['libraries_pred'].apply(lambda x: ast.literal_eval(x))
# df_pos['gt_new'] = df_pos['gt_new'].apply(lambda x: ast.literal_eval(x))
# df_pos['pred'] = df_pos['gt_new'].apply(lambda x:x[1])
# print(df_pos['pred'])
# print(df_pos['libraries_pred'])
# df_pos['disagree'] = df_pos[['libraries_pred', 'pred']].apply(lambda x: disagree(x[0], x[1]), axis=1)
# print(df_pos['disagree'])
#
# df_pos_exp = df_pos.explode('disagree')
# df_pos['disagree_together'] = df_pos[['libraries_pred', 'pred']].apply(lambda x: disagree_together(x[0], x[1]), axis=1)
# print(df_pos['disagree_together'])
#
#
# print(df_pos_exp['gt_new'])
# df_pos_exp = df_pos.explode('disagree')
# ax = sns.countplot(x="disagree", data=df_pos_exp)
# plt.show()
#
# df_pos_exp2 = df_pos.explode('disagree_together')
# ax = sns.countplot(x = 'disagree_together',
#               data = df_pos_exp2,
#               order = df_pos_exp2['disagree_together'].value_counts().index)
# plt.xticks(rotation=90)
# plt.show()
#
#
# for i in [2, 3, 4, 5]:
#     df_pos = pd.read_csv('sentences_to_GT_POS_corr_stats_new_gt.csv')
#     df_pos = df_pos[df_pos.count_most_frequent == i]
#     df_pos['libraries_pred'] = df_pos['libraries_pred'].apply(lambda x: ast.literal_eval(x))
#     df_pos['gt_new'] = df_pos['gt_new'].apply(lambda x: ast.literal_eval(x))
#     df_pos['pred'] = df_pos['gt_new'].apply(lambda x: x[2])
#     print(df_pos['pred'])
#     print(df_pos['libraries_pred'])
#     df_pos['disagree'] = df_pos[['libraries_pred', 'pred']].apply(lambda x: disagree(x[0], x[1]), axis=1)
#     print(df_pos['disagree'])
#
#     df_pos_exp = df_pos.explode('disagree')
#     df_pos['disagree_together'] = df_pos[['libraries_pred', 'pred']].apply(lambda x: disagree_together(x[0], x[1]),
#                                                                            axis=1)
#     print(df_pos['disagree_together'])
#
#     print(df_pos_exp['gt_new'])
#     df_pos_exp = df_pos.explode('disagree')
#     ax = sns.countplot(x="disagree", data=df_pos_exp)
#     plt.title('{}'.format(i))
#     plt.show()
#
#     df_pos_exp2 = df_pos.explode('disagree_together')
#     ax = sns.countplot(x='disagree_together',
#                        data=df_pos_exp2,
#                        order=df_pos_exp2['disagree_together'].value_counts().index)
#     plt.xticks(rotation=90)
#     plt.title('{}'.format(i))
#     plt.show()


df_pos = pd.read_csv('sentences_to_GT_POS_corr_stats_new_gt.csv')
df_pos[(df_pos.count_most_frequent == 5) & (df_pos.is_most_frequent_gt == 0)].to_csv('temp5.csv')
print(len(df_pos[(df_pos.count_most_frequent == 5) & (df_pos.is_most_frequent_gt == 0)]))
