import pandas as pd
from collections import Counter
import seaborn as sns

df_pos = pd.read_csv('sentences_to_GT_POS_corr_temp.csv')

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

df_pos['gt_new'] = df_pos['sentence']


def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]


for i in range(len(df_pos)):
    list_tokens = []
    for tok, gt, index, val_nltk, val_stanza, val_textblob, val_spacy, val_gc in zip(df_pos['sentence_tok'].tolist(),
                                                                                     df_pos['GT'].tolist(),
                                                                                     df_pos['gt_index'].tolist(),
                                                                                     df_pos['pos_nltk_univ'].tolist(),
                                                                                     df_pos['pos_stanza_univ'].tolist(),
                                                                                     df_pos[
                                                                                         'pos_textblob_univ'].tolist(),
                                                                                     df_pos['pos_spacy_univ'].tolist(),
                                                                                     df_pos['pos_gc_univ'].tolist()):
        if index in df_pos['spacy_index'].tolist() + df_pos['stanza_index'].tolist() + df_pos['gc_index'].tolist() + \
                df_pos[
                    'nltk_index'].tolist() + df_pos['textblob_index'].tolist():
            most_frequent = most_frequent([gt, val_gc, val_nltk, val_stanza, val_spacy, val_textblob])
            most_frequent_stanza_gc_overrated = most_frequent(
                [gt, val_gc, val_gc, val_nltk, val_stanza, val_stanza, val_spacy, val_textblob])
            df_pos.loc[i, 'gt_new'] = list_tokens.append(
                [tok, gt, most_frequent, [gt, val_gc, val_nltk, val_stanza, val_spacy, val_textblob].count(most_frequent),
                 len(list(set([gt, val_gc, val_nltk, val_stanza, val_spacy, val_textblob]))),
                 most_frequent == gt, most_frequent_stanza_gc_overrated,
                 [gt, val_gc, val_nltk, val_stanza, val_spacy, val_textblob].count(most_frequent_stanza_gc_overrated),
                 most_frequent_stanza_gc_overrated == gt])

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
sns_plot4 = sns.kdeplot(data=df_pos_exp[df_pos_exp.is_most_frequent_gt==1], x="most_frequent", hue="gt")
sns_plot4.savefig("output4.png")
sns_plot5 = sns.kdeplot(data=df_pos_exp[df_pos_exp.is_most_frequent_gt==1], x="count_most_frequent", hue="gt")
sns_plot5.savefig("output5.png")