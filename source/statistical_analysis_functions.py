import pandas as pd


def create_df_all_sentences():
    """create the df with pos tags given by each libraries, for each sentence"""
    # df_pos = pd.read_csv(os.path.join(THIS_FOLDER, 'source/utils/sentences_to_GT_POS.csv'))
    # gt = split_labels_articles_that_need_to(df_pos[df_pos.sentence == sent]['tagged_tokens_GT'].apply(
    #     lambda x: [i[1] for i in ast.literal_eval(x)]).tolist()[0])
    # sent_tok = _split_composite_pos_tokens(df_pos[df_pos.sentence == sent]['tagged_tokens_GT'].apply(
    #     lambda x: [i[0] for i in ast.literal_eval(x)]).tolist()[0])
    # return [(sent_tok_val, gt_val) for gt_val, sent_tok_val in zip(gt, sent_tok)]


def df_new_column_with_votes_statistics():
    """check votes statistics for each token, so the majority tag, how many times its voted, the number of unique tags,
     wether the manjority tag equals the GT and the libraries predictions"""


def df_tokens_4_agree_and_different_GT():
    """when 4 libraries agree and not the GT we check wether we can trust the libraries"""


def df_tokens_3_agree_2_unique_and_different_GT():
    """when 3 libraries agree and not the GT which agrees with an other library"""


def chart1():
    """when 3 libraries agree and not the GT which agrees with an other library, which library it tends to be?"""


def df_tokens_3_agree_3_unique_and_different_GT():
    """when 3 libraries agree and not the GT, and an other library is voting for an other tag"""


def df_tokens_3_agree_2_unique_and_same_GT():
    """when 3 libraries agree, the GT as well and 2 other libraries agree for an other tag"""


def chart2():
    """when 3 libraries agree, the GT as well and 2 other libraries agree for an other tag, do we have 2 libraries
    agreeing together that stand out?"""


def df_tokens_3_agree_3_unique_and_same_GT():
    """when 3 libraries agree, the GT as well and 2 other libraries are voting for 2 other tags"""


def df_tokens_2_agree_4_unique_and_different_GT():
    """when 2 libraries agree and not the GT, and 2 remaining libraries are voting for other different tags"""


def df_tokens_2_agree_3_unique_and_different_GT():
    """when 2 libraries agree and not the GT, and 2 other votes are the same"""


def chart3():
    """when 2 libraries agree and not the GT, and 2 other votes are the same, where do they tend to come from?"""


def df_tokens_2_agree_4_unique_and_same_GT():
    """when 3 libraries agree, the GT as well and 3 other libraries are voting for 3 other different tags"""


def df_tokens_2_agree_3_unique_and_same_GT():
    """when 2 libraries agree, the GT as well and 2 remaining libraries are voting for 2 other same tags"""


def chart3():
    """when 2 libraries agree, the GT as well and 2 remaining libraries are voting for 2 other same tags, what tend to be these libraries?"""


def df_tokens_1_agree():
    """here all libraries vote for different pos tags, see wether we keep the GT, we change it to an other library
    pos tag that we trust more (be careful to final bias) or we mnually review it"""
