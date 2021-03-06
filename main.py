"""
Script for running pos tagging and comparing results.
"""

from source.statistical_analysis_functions import *
from source.eval.pos_tagging_evaluation import evaluation

if __name__ == "__main__":
    # create_df_all_sentences()
    # df_new_column_with_votes_statistics()
    # df_tokens_4_agree_and_different_GT()
    # df_tokens_4_agree_and_same_GT()
    # chart_4_agree_same_GT_what_is_the_other_library()
    # df_tokens_3_agree_2_unique_and_different_GT()
    # chart_3_agree_different_GT_which_library_GT_tends_to_agree_with()
    # df_tokens_3_agree_3_unique_and_different_GT()
    # df_tokens_3_agree_2_unique_and_same_GT()
    # chart_3_agree_same_GT_which_libraries_tends_to_agree_with_each_other()
    # df_tokens_3_agree_3_unique_and_same_GT()
    # df_tokens_2_agree_4_unique_and_different_GT()
    # df_tokens_2_agree_3_unique_and_different_GT()
    # chart_2_agree_different_GT_which_library_GT_tends_to_agree_with()
    # df_tokens_2_agree_4_unique_and_same_GT()
    # df_tokens_2_agree_3_unique_and_same_GT()
    # chart_2_agree_same_GT_which_libraries_tends_to_agree_with_each_other()
    # df_tokens_1_agree()
    # revewing_dataset()

    evaluation('source/eval/sentences_to_GT_POS_corrected.csv')