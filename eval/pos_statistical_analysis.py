import pandas as pd
from collections import Counter
import seaborn as sns
import numpy as np
import ast
import matplotlib.pyplot as plt

# df_pos = pd.read_csv('sentences_to_GT_POS_corr_temp.csv')
#
# df_pos['gt_new'] = df_pos['sentence']
#
# def most_frequent(List):
#     occurence_count = Counter(List)
#     return occurence_count.most_common(1)[0][0]
#
#
# for i in range(len(df_pos)):
#     list_tokens = []
#     for tok, gt, index, val_nltk, val_stanza, val_textblob, val_spacy, val_gc in zip(ast.literal_eval(df_pos.loc[i, 'sentence_tok']),
#                                                                                      ast.literal_eval(df_pos.loc[i, 'GT']),
#                                                                                      ast.literal_eval(df_pos.loc[i, 'GT_index']),
#                                                                                      ast.literal_eval(df_pos.loc[i, 'pos_nltk_univ']),
#                                                                                      ast.literal_eval(df_pos.loc[i, 'pos_stanza_univ']),
#                                                                                      ast.literal_eval(df_pos.loc[i,
#                                                                                     'pos_textblob_univ']),
#                                                                                      ast.literal_eval(df_pos.loc[i, 'pos_spacy_univ']),
#                                                                                      ast.literal_eval(df_pos.loc[i, 'pos_gc_univ'])):
#         if index in ast.literal_eval(df_pos.loc[i, 'spacy_index']) + ast.literal_eval(df_pos.loc[i, 'stanza_index']) + ast.literal_eval(df_pos.loc[i, 'gc_index']) + \
#                 ast.literal_eval(df_pos.loc[i, 'nltk_index']) + ast.literal_eval(df_pos.loc[i, 'textblob_index']):
#             most_frequent_val = most_frequent([gt, val_gc, val_nltk, val_stanza, val_spacy, val_textblob])
#             most_frequent_val_stanza_gc_overrated = most_frequent(
#                 [gt, val_gc, val_gc, val_nltk, val_stanza, val_stanza, val_spacy, val_textblob])
#             list_tokens.append(
#                 [tok, gt, most_frequent_val, [gt, val_gc, val_nltk, val_stanza, val_spacy, val_textblob].count(most_frequent_val),
#                  len(list(set([gt, val_gc, val_nltk, val_stanza, val_spacy, val_textblob]))),
#                  most_frequent_val == gt, most_frequent_val_stanza_gc_overrated,
#                  [gt, val_gc, val_nltk, val_stanza, val_spacy, val_textblob].count(most_frequent_val_stanza_gc_overrated),
#                  most_frequent_val_stanza_gc_overrated == gt])
#     df_pos.loc[i, 'gt_new'] = str(list_tokens)
#
# print(df_pos['gt_new'])
# df_pos.to_csv('sentences_to_GT_POS_corr_stats.csv')
#
# df_pos = pd.read_csv('sentences_to_GT_POS_corr_stats.csv')
#
# df_pos['gt_new'] = df_pos['gt_new'].apply(lambda x: ast.literal_eval(x))
# df_pos_exp = df_pos.explode('gt_new')
# df_pos_exp['is_gt_new_list'] = df_pos_exp['gt_new'].apply(lambda x: type(x))
# df_pos_exp = df_pos_exp[df_pos_exp.is_gt_new_list == list]
# df_pos_exp[['tok', 'gt', 'most_frequent', 'count_most_frequent', 'count_uniques', 'is_most_frequent_gt',
#             'most_frequent_stanza_gt_higher_weights', 'count_most_frequent_stanza_gt_hw',
#             'is_most_frequent_gt_hw']] = pd.DataFrame(df_pos_exp.gt_new.tolist(), index=df_pos_exp.index)
#
# df_pos_exp.to_csv('sentences_to_GT_POS_corr_stats2.csv')


df_pos_exp = pd.read_csv('sentences_to_GT_POS_corr_stats2.csv')
print(df_pos_exp[df_pos_exp.is_most_frequent_gt==1])
sns.kdeplot(data=df_pos_exp, x="count_most_frequent", hue="is_most_frequent_gt")
plt.title('When a major vote equals the GT or not, distributions of counting votes')
plt.show()
sns.kdeplot(data=df_pos_exp, x="count_most_frequent", hue="most_frequent")
plt.title('How libraries tend to agree for each tag,  distributions of counting votes')
plt.show()
sns.kdeplot(data=df_pos_exp[df_pos_exp.is_most_frequent_gt==1], x="count_most_frequent", hue="gt")
plt.title('How libraries tend to agree with the GT, distribution of counting votes')
plt.show()

f, ax = plt.subplots(figsize=(16, 12))
df_dummies = pd.get_dummies(df_pos_exp[['most_frequent', 'gt']])
print(df_dummies)

x = df_dummies.values
correlation_matrix = np.corrcoef(x.T)
print(correlation_matrix)

# plot the heatmap
sns.heatmap(correlation_matrix,
            xticklabels=df_dummies.columns,
            yticklabels=df_dummies.columns)
plt.show()

