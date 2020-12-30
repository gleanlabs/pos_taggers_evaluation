from files.helper_maps import *

import json
import pandas as pd
import nltk
from tokenizer.treebankwordtokenizer import RevisedTreeBankWordTokenizer
from tokenizer.treebankwordtokenizer_spacy_whitespace import RevisedTreeBankWordTokenizerVocab
import stanza
import en_core_web_sm
# stanza.download('en')
import spacy
from sklearn.metrics import precision_score

spacy.load('en_core_web_sm')
import numpy as np


def transform_golden(istring, start_index=1):
	ifile = open(istring, 'r')
	lines_list = []
	added_so_far = start_index
	for line in ifile:
		line = line.rstrip()  # remove \n from the end of the line

		elem = line.split('\t')
		tokens = elem[0].split(' ')  # todo: may not be needed ( is it used even?)
		tokens_pos = elem[1].split(' ')
		sentence = elem[0]

		lines_list.append((added_so_far, sentence, tokens, tokens_pos))
		added_so_far += 1
	return lines_list


def article_to_universal(GT_pos):
	"""

	:param GT_pos: The POS we get from GroundTruth (not in PTB)
	:return: The POS we get from GroundTruth (in PTB) if the translation exists, "UNK" (unknown) otherwise.
	"""
	if GT_pos in ARTICLE_TO_UNIVERSAL_MAP:
		return ARTICLE_TO_UNIVERSAL_MAP[GT_pos]
	else:
		return "[UNK]"  # unknown


def stage_1_table_creation(result_dict_so_far, input_path, start_index=1):
	""" GT to universal + NLTK POS tagging to universal """
	sentences_info = transform_golden(istring=input_path, start_index=start_index)

	tokenizer = RevisedTreeBankWordTokenizer()
	for sent_info in sentences_info:
		sent_num, sentence, sent_gt_articlt_pos = sent_info[0], sent_info[1], sent_info[3]
		sentence_tok = tokenizer.tokenize(sentence)
		pos_nltk = [pos[1] for pos in nltk.pos_tag(sentence_tok)]

		result_dict_so_far[sent_num] = {"tokens": sentence_tok,
										"sentence": sentence,
										"nltk_pos": pos_nltk,
										"GT_pos": [article_to_universal(article_pos) for
												   article_pos in sent_gt_articlt_pos]}
	return result_dict_so_far


def stage_2_adding_spacy_POS(result_dict_so_far):
	""" SPACY pos tagging to universal"""
	nlp_spacy = en_core_web_sm.load()
	nlp_spacy.tokenizer = RevisedTreeBankWordTokenizerVocab(nlp_spacy.vocab)

	for k in result_dict_so_far.keys():
		sentence = result_dict_so_far[k]["sentence"]
		sentence_spacy = nlp_spacy(sentence)
		pos_spacy = [token.pos_ for token in sentence_spacy]
		pos_spacy_univ = [UNIVERSAL_MAP[pos] if pos in UNIVERSAL_MAP else pos for pos in pos_spacy]
		result_dict_so_far[k]["spacy_POS_universal"] = pos_spacy_univ
	return result_dict_so_far


def stage_3_adding_stanza_POS(result_dict_so_far):
	""" stanza POS tagging to universal"""
	nlp_stanza = stanza.Pipeline(lang='en', processors='tokenize,pos', tokenize_pretokenized=True)

	for k in result_dict_so_far.keys():
		tokens = [result_dict_so_far[k]["tokens"]]
		sentences_stanza = nlp_stanza(tokens)  # the same tokens from stage 1 ( for the current sentence )
		# sent_pos_stanza = [word.xpos for word in sentences_stanza.words]
		sent_pos_stanza = [[word.xpos for word in s.words] for s in sentences_stanza.sentences]
		sent_pos_stanza = sent_pos_stanza[0]

		result_dict_so_far[k]["stanza_POS"] = [PENN_TREEBANK_TO_UNIVERSAL_MAP[pos]
											   if pos in PENN_TREEBANK_TO_UNIVERSAL_MAP
											   else "[UNK]" for pos in sent_pos_stanza]
	return result_dict_so_far

def main():
	# stage 1
	file_paths = [r"C:\Users\admin\Desktop\newly_annotated_data\golden-cy.txt",
				  r"C:\Users\admin\Desktop\newly_annotated_data\golden-lf.txt",
				  r"C:\Users\admin\Desktop\newly_annotated_data\golden-lj.txt",
				  r"C:\Users\admin\Desktop\newly_annotated_data\golden-ly.txt",
				  r"C:\Users\admin\Desktop\newly_annotated_data\golden-wt.txt",
				  r"C:\Users\admin\Desktop\newly_annotated_data\golden-zc.txt"
				  ]
	result_dict_so_far = {}
	for path in file_paths:
		stage_1_table_creation(result_dict_so_far, input_path=path, start_index=len(result_dict_so_far) + 1)

	# stage 2
	stage_2_adding_spacy_POS(result_dict_so_far)

	# stage 3
	stage_3_adding_stanza_POS(result_dict_so_far)

	write_to_json(result_dict_so_far, output_file_name="stage_3_output")

	return result_dict_so_far


def create_csv_GT_table(list_of_tuples, output_path="sentences_to_GT_POS_corrected_Basel.csv", list_of_column_names=['Sentence', 'GT_POS'], index_name=""):
	# make groundtruth dataframe
	df = pd.DataFrame(list_of_tuples, columns=list_of_column_names)
	if index_name:
		df.index.name = index_name
	df.to_csv(output_path)
	return df


def write_to_json(dict_to_write, output_file_name='output'):
	with open('{}.json'.format(output_file_name), 'w') as fp:
		# # pretty print
		# s = json.dumps(dict_to_write, indent=0, sort_keys=True)
		# fp.write(s)
		json.dump(dict_to_write, fp)


if __name__ == "__main__":
	result_dict = main()
	sentences_to_gt_pos = [(result_dict[k]["sentence"], result_dict[k]["GT_pos"]) for k in result_dict]
	create_csv_GT_table(sentences_to_gt_pos)
