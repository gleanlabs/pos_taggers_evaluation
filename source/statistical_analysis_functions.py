import pandas as pd
import os
from source.tag_pos import _pos_tag_sentence, map_results_to_universal_tags
from source.tag_statistics import *

THIS_FOLDER = "/Users/johanna/Desktop/pos_taggers_evaluation/pos_taggers_evaluation/"
LIST_PACKAGES = ['nltk', 'stanza', 'spacy', 'flair', 'article']


def create_df_all_sentences():
    """create the df with pos tags given by each libraries, for each sentence"""
    df_pos = pd.read_csv(os.path.join(THIS_FOLDER, 'source/utils/sentences_to_GT_POS.csv'))
    for index, doc in enumerate(df_pos['sentence'].tolist()):
        print(index)
        for lib in LIST_PACKAGES:
            df_pos.loc[index, lib + '_pos'] = str(map_results_to_universal_tags(_pos_tag_sentence(lib, doc), lib))
    df_pos.to_csv(os.path.join(THIS_FOLDER, 'source/utils/sentences_to_GT_POS_libraries.csv'))


def df_new_column_with_votes_statistics():
    """check votes statistics for each token, so the majority tag, how many times its voted, the number of unique tags,
     wether the manjority tag equals the GT and the libraries predictions"""
    df_pos = pd.read_csv(os.path.join(THIS_FOLDER, 'source/utils/sentences_to_GT_POS_libraries.csv'))
    df_pos['token_votes'] = df_pos[[lib + '_pos' for lib in LIST_PACKAGES]].apply(
        lambda x: [
            {'token': list_token_tags[0][0], 'GT': list_token_tags[-1][1], 'votes': list_token_tags,
             'majority_token': return_majority_token(list_token_tags),
             'nb_votes_majority_token': return_number_votes_majority_token(
                 list_token_tags), 'unique_tokens': return_unique_tokens(list_token_tags),
             'is_majority_token_equals_gt': return_wether_majority_token_equals_gt(list_token_tags)} for list_token_tags
            in
            zip(*x)], axis=1)
    df_pos_exp = df_pos.explode('token_votes')
    df_pos_exp.to_csv(os.path.join(THIS_FOLDER, 'source/utils/sentences_to_GT_POS_libraries_votes_exp.csv'))


def df_tokens_4_agree_and_different_GT():
    """when 4 libraries agree and not the GT we check wether we can trust the libraries"""
    df_pos_exp = pd.read_csv(os.path.join(THIS_FOLDER, 'source/utils/sentences_to_GT_POS_libraries_votes_exp.csv'))
    df_pos_exp['nb_votes_majority_token'] = df_pos_exp['token_votes'].apply(lambda x: x['nb_votes_majority_token'])
    df_pos_exp['equals_GT'] = df_pos_exp['token_votes'].apply(lambda x: x['is_majority_token_equals_gt'])
    df_pos_exp[
        (df_pos_exp['nb_votes_majority_token'] == 4)(df_pos_exp['equals_GT'] == 0)].to_csv(
        os.path.join(THIS_FOLDER, 'source/utils/sentences_4_agree_different_GT.csv'))


def df_tokens_3_agree_2_unique_and_different_GT():
    """when 3 libraries agree and not the GT which agrees with an other library"""
    df_pos_exp = pd.read_csv(os.path.join(THIS_FOLDER, 'source/utils/sentences_to_GT_POS_libraries_votes_exp.csv'))
    df_pos_exp['nb_votes_majority_token'] = df_pos_exp['token_votes'].apply(lambda x: x['nb_votes_majority_token'])
    df_pos_exp['unique_tokens'] = df_pos_exp['token_votes'].apply(lambda x: x['unique_tokens'])
    df_pos_exp['equals_GT'] = df_pos_exp['token_votes'].apply(lambda x: x['is_majority_token_equals_gt'])
    df_pos_exp[
        (df_pos_exp['nb_votes_majority_token'] == 3) & (df_pos_exp['unique_tokens'] == 2) & (
                df_pos_exp['equals_GT'] == 0)].to_csv(
        os.path.join(THIS_FOLDER, 'source/utils/sentences_3_agree_2_unique_different_GT.csv'))


# TODO: rename the chart
# TODO: make a readme with charts as well
# TODO: write a recommendation part for later use

def chart_3_agree_different_GT_which_library_GT_tends_to_agree_with():
    """when 3 libraries agree and not the GT which agrees with an other library, which library it tends to be?"""
    df_pos_exp = pd.read_csv(os.path.join(THIS_FOLDER, 'source/utils/sentences_3_agree_2_unique_different_GT.csv'))


def df_tokens_3_agree_3_unique_and_different_GT():
    """when 3 libraries agree and not the GT, and an other library is voting for an other tag"""
    df_pos_exp = pd.read_csv(os.path.join(THIS_FOLDER, 'source/utils/sentences_to_GT_POS_libraries_votes_exp.csv'))
    df_pos_exp['nb_votes_majority_token'] = df_pos_exp['token_votes'].apply(lambda x: x['nb_votes_majority_token'])
    df_pos_exp['unique_tokens'] = df_pos_exp['token_votes'].apply(lambda x: x['unique_tokens'])
    df_pos_exp['equals_GT'] = df_pos_exp['token_votes'].apply(lambda x: x['is_majority_token_equals_gt'])
    df_pos_exp[
        (df_pos_exp['nb_votes_majority_token'] == 3) & (df_pos_exp['unique_tokens'] == 3) & (
                df_pos_exp['equals_GT'] == 0)].to_csv(
        os.path.join(THIS_FOLDER, 'source/utils/sentences_3_agree_3_unique_different_GT.csv'))


def df_tokens_3_agree_2_unique_and_same_GT():
    """when 3 libraries agree, the GT as well and 2 other libraries agree for an other tag"""
    df_pos_exp = pd.read_csv(os.path.join(THIS_FOLDER, 'source/utils/sentences_to_GT_POS_libraries_votes_exp.csv'))
    df_pos_exp['nb_votes_majority_token'] = df_pos_exp['token_votes'].apply(lambda x: x['nb_votes_majority_token'])
    df_pos_exp['unique_tokens'] = df_pos_exp['token_votes'].apply(lambda x: x['unique_tokens'])
    df_pos_exp['equals_GT'] = df_pos_exp['token_votes'].apply(lambda x: x['is_majority_token_equals_gt'])
    df_pos_exp[
        (df_pos_exp['nb_votes_majority_token'] == 3) & (df_pos_exp['unique_tokens'] == 2) & (
                df_pos_exp['equals_GT'] == 1)].to_csv(
        os.path.join(THIS_FOLDER, 'source/utils/sentences_3_agree_2_unique_same_GT.csv'))


def chart_3_agree_same_GT_which_libraries_tends_to_agree_with_each_other():
    """when 3 libraries agree, the GT as well and 2 other libraries agree for an other tag, do we have 2 libraries
    agreeing together that stand out?"""
    df_pos_exp = pd.read_csv(os.path.join(THIS_FOLDER, 'source/utils/sentences_3_agree_2_unique_same_GT.csv'))


def df_tokens_3_agree_3_unique_and_same_GT():
    """when 3 libraries agree, the GT as well and 2 other libraries are voting for 2 other tags"""
    df_pos_exp = pd.read_csv(os.path.join(THIS_FOLDER, 'source/utils/sentences_to_GT_POS_libraries_votes_exp.csv'))
    df_pos_exp['nb_votes_majority_token'] = df_pos_exp['token_votes'].apply(lambda x: x['nb_votes_majority_token'])
    df_pos_exp['unique_tokens'] = df_pos_exp['token_votes'].apply(lambda x: x['unique_tokens'])
    df_pos_exp['equals_GT'] = df_pos_exp['token_votes'].apply(lambda x: x['is_majority_token_equals_gt'])
    df_pos_exp[
        (df_pos_exp['nb_votes_majority_token'] == 3) & (df_pos_exp['unique_tokens'] == 3) & (
                df_pos_exp['equals_GT'] == 1)].to_csv(
        os.path.join(THIS_FOLDER, 'source/utils/sentences_3_agree_3_unique_same_GT.csv'))


def df_tokens_2_agree_4_unique_and_different_GT():
    """when 2 libraries agree and not the GT, and 2 remaining libraries are voting for other different tags"""
    df_pos_exp = pd.read_csv(os.path.join(THIS_FOLDER, 'source/utils/sentences_to_GT_POS_libraries_votes_exp.csv'))
    df_pos_exp['nb_votes_majority_token'] = df_pos_exp['token_votes'].apply(lambda x: x['nb_votes_majority_token'])
    df_pos_exp['unique_tokens'] = df_pos_exp['token_votes'].apply(lambda x: x['unique_tokens'])
    df_pos_exp['equals_GT'] = df_pos_exp['token_votes'].apply(lambda x: x['is_majority_token_equals_gt'])
    df_pos_exp[
        (df_pos_exp['nb_votes_majority_token'] == 2) & (df_pos_exp['unique_tokens'] == 4) & (
                df_pos_exp['equals_GT'] == 0)].to_csv(
        os.path.join(THIS_FOLDER, 'source/utils/sentences_2_agree_4_unique_different_GT.csv'))


def df_tokens_2_agree_3_unique_and_different_GT():
    """when 2 libraries agree and not the GT, and 2 other votes are the same"""
    df_pos_exp = pd.read_csv(os.path.join(THIS_FOLDER, 'source/utils/sentences_to_GT_POS_libraries_votes_exp.csv'))
    df_pos_exp['nb_votes_majority_token'] = df_pos_exp['token_votes'].apply(lambda x: x['nb_votes_majority_token'])
    df_pos_exp['unique_tokens'] = df_pos_exp['token_votes'].apply(lambda x: x['unique_tokens'])
    df_pos_exp['equals_GT'] = df_pos_exp['token_votes'].apply(lambda x: x['is_majority_token_equals_gt'])
    df_pos_exp[
        (df_pos_exp['nb_votes_majority_token'] == 2) & (df_pos_exp['unique_tokens'] == 3) & (
                df_pos_exp['equals_GT'] == 0)].to_csv(
        os.path.join(THIS_FOLDER, 'source/utils/sentences_2_agree_3_unique_different_GT.csv'))


def chart_2_agree_different_GT_which_library_GT_tends_to_agree_with():
    """when 2 libraries agree and not the GT, and 2 other votes are the same, where do they tend to come from?"""
    df_pos_exp = pd.read_csv(os.path.join(THIS_FOLDER, 'source/utils/sentences_2_agree_3_unique_different_GT.csv'))


def df_tokens_2_agree_4_unique_and_same_GT():
    """when 3 libraries agree, the GT as well and 3 other libraries are voting for 3 other different tags"""
    df_pos_exp = pd.read_csv(os.path.join(THIS_FOLDER, 'source/utils/sentences_to_GT_POS_libraries_votes_exp.csv'))
    df_pos_exp['nb_votes_majority_token'] = df_pos_exp['token_votes'].apply(lambda x: x['nb_votes_majority_token'])
    df_pos_exp['unique_tokens'] = df_pos_exp['token_votes'].apply(lambda x: x['unique_tokens'])
    df_pos_exp['equals_GT'] = df_pos_exp['token_votes'].apply(lambda x: x['is_majority_token_equals_gt'])
    df_pos_exp[
        (df_pos_exp['nb_votes_majority_token'] == 2) & (df_pos_exp['unique_tokens'] == 4) & (
                df_pos_exp['equals_GT'] == 1)].to_csv(
        os.path.join(THIS_FOLDER, 'source/utils/sentences_2_agree_4_unique_same_GT.csv'))


def df_tokens_2_agree_3_unique_and_same_GT():
    """when 2 libraries agree, the GT as well and 2 remaining libraries are voting for 2 other same tags"""
    df_pos_exp = pd.read_csv(os.path.join(THIS_FOLDER, 'source/utils/sentences_to_GT_POS_libraries_votes_exp.csv'))
    df_pos_exp['nb_votes_majority_token'] = df_pos_exp['token_votes'].apply(lambda x: x['nb_votes_majority_token'])
    df_pos_exp['unique_tokens'] = df_pos_exp['token_votes'].apply(lambda x: x['unique_tokens'])
    df_pos_exp['equals_GT'] = df_pos_exp['token_votes'].apply(lambda x: x['is_majority_token_equals_gt'])
    df_pos_exp[
        (df_pos_exp['nb_votes_majority_token'] == 2) & (df_pos_exp['unique_tokens'] == 3) & (
                df_pos_exp['equals_GT'] == 1)].to_csv(
        os.path.join(THIS_FOLDER, 'source/utils/sentences_2_agree_3_unique_same_GT.csv'))


def chart_2_agree_same_GT_which_libraries_tends_to_agree_with_each_other():
    """when 2 libraries agree, the GT as well and 2 remaining libraries are voting for 2 other same tags, what tend to be these libraries?"""
    df_pos_exp = pd.read_csv(os.path.join(THIS_FOLDER, 'source/utils/sentences_2_agree_3_unique_same_GT.csv'))


def df_tokens_1_agree():
    """here all libraries vote for different pos tags, see wether we keep the GT, we change it to an other library
    pos tag that we trust more (be careful to final bias) or we mnually review it"""
    df_pos_exp = pd.read_csv(os.path.join(THIS_FOLDER, 'source/utils/sentences_to_GT_POS_libraries_votes_exp.csv'))
    df_pos_exp['nb_votes_majority_token'] = df_pos_exp['token_votes'].apply(lambda x: x['nb_votes_majority_token'])
    df_pos_exp[
        (df_pos_exp['nb_votes_majority_token'] == 1)].to_csv(
        os.path.join(THIS_FOLDER, 'source/utils/sentences_1_agree.csv'))
