import pandas as pd
from collections import Counter
import seaborn as sns
import numpy as np
import ast
df_pos = pd.read_csv('sentences_to_GT_POS_corr_temp4.csv')

df_pos['gt_new'] = df_pos['sentence']

def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]


for i in range(len(df_pos)):
    list_tokens = []
    for tok, gt, index, val_nltk, val_stanza, val_textblob, val_spacy, val_gc in zip(ast.literal_eval(df_pos.loc[i, 'sentence_tok']),
                                                                                     ast.literal_eval(df_pos.loc[i, 'GT']),
                                                                                     ast.literal_eval(df_pos.loc[i, 'GT_index']),
                                                                                     ast.literal_eval(df_pos.loc[i, 'pos_nltk_univ']),
                                                                                     ast.literal_eval(df_pos.loc[i, 'pos_stanza_univ']),
                                                                                     ast.literal_eval(df_pos.loc[i,
                                                                                    'pos_textblob_univ']),
                                                                                     ast.literal_eval(df_pos.loc[i, 'pos_spacy_univ']),
                                                                                     ast.literal_eval(df_pos.loc[i, 'pos_gc_univ'])):
        if index in ast.literal_eval(df_pos.loc[i, 'spacy_index']) + ast.literal_eval(df_pos.loc[i, 'stanza_index']) + ast.literal_eval(df_pos.loc[i, 'gc_index']) + \
                ast.literal_eval(df_pos.loc[i, 'nltk_index']) + ast.literal_eval(df_pos.loc[i, 'textblob_index']):
            most_frequent_val = most_frequent([gt, val_gc, val_nltk, val_stanza, val_spacy, val_textblob])
            most_frequent_val_stanza_gc_overrated = most_frequent(
                [gt, val_gc, val_gc, val_nltk, val_stanza, val_stanza, val_spacy, val_textblob])
            list_tokens.append(
                [tok, gt, most_frequent_val, [gt, val_gc, val_nltk, val_stanza, val_spacy, val_textblob].count(most_frequent_val),
                 len(list(set([gt, val_gc, val_nltk, val_stanza, val_spacy, val_textblob]))),
                 most_frequent_val == gt, most_frequent_val_stanza_gc_overrated,
                 [gt, val_gc, val_nltk, val_stanza, val_spacy, val_textblob].count(most_frequent_val_stanza_gc_overrated),
                 most_frequent_val_stanza_gc_overrated == gt])
    df_pos.loc[i, 'gt_new'] = str(list_tokens)

print(df_pos['gt_new'])

df_pos.to_csv('sentences_to_GT_POS_corr_stats.csv')

df_pos_exp = df_pos.explode('gt_new')
df_pos_exp[['tok', 'gt', 'most_frequent', 'count_most_frequent', 'count_uniques', 'is_most_frequent_gt',
            'most_frequent_stanza_gt_higher_weights', 'count_most_frequent_stanza_gt_hw',
            'is_most_frequent_gt_hw']] = pd.DataFrame(df_pos_exp.gt_new.tolist(), index=df_pos_exp.index)
sns_plot1 = sns.kdeplot(data=df_pos_exp, x="count_most_frequent", hue="is_most_frequent_gt")
sns_plot1.savefig("output1.png")
sns_plot2 = sns.kdeplot(data=df_pos_exp, x="count_most_frequent", hue="most_frequent")
sns_plot2.savefig("output2.png")
sns_plot3 = sns.kdeplot(data=df_pos_exp[df_pos_exp.is_most_frequent_gt==0], x="most_frequent", hue="gt")
sns_plot3.savefig("output3.png")
sns_plot4 = sns.kdeplot(data=df_pos_exp[df_pos_exp.is_most_frequent_gt==1], x="count_uniques", hue="gt")
sns_plot4.savefig("output4.png")
sns_plot5 = sns.kdeplot(data=df_pos_exp[df_pos_exp.is_most_frequent_gt==1], x="count_most_frequent", hue="gt")
sns_plot5.savefig("output5.png")


#heatmap things appearing together
import matplotlib.pyplot as plt

sns.set_theme(style="white")

# Compute the correlation matrix
corr = df_pos.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})




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