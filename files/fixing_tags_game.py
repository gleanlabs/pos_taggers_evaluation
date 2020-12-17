from files.helper_maps import ARTICLE_TO_HUMAN_MAP
from files.new_funcs_updated import write_to_json, transform_golden, create_csv_GT_table


def get_dataset_from_files(list_of_GT_paths):
	"""

	:param list_of_GT_paths: list of file paths tp read  from
	:return:
	"""
	rows = []
	for input_path in list_of_GT_paths:
		sentences_info = transform_golden(istring=input_path, start_index=len(rows) + 1)
		rows.extend(sentences_info)
	# prepare outputs
	output_columns = []
	output_column_names = ["sentence", "tagged_tokens_GT", "tagged_tokens_GT_corrected"]
	for added_so_far, sentence, tokens, tokens_pos in rows:
		token_to_tag_list = list(zip(tokens, tokens_pos))
		output_columns.append((sentence, token_to_tag_list, token_to_tag_list))
	# write_to_output_file
	create_csv_GT_table(list_of_tuples=output_columns, list_of_column_names=output_column_names, index_name="row")


def get_dataset_from_files_alternative(list_of_GT_paths):
	"""

	:param list_of_GT_paths: list of file paths tp read  from
	:return:
	"""
	rows = []
	for input_path in list_of_GT_paths:
		sentences_info = transform_golden(istring=input_path, start_index=len(rows) + 1)
		rows.extend(sentences_info)
	# prepare outputs
	output_columns = []
	output_column_names = ["sent_id", "sentence", "tagged_tokens_GT"]
	for added_so_far, sentence, tokens, tokens_pos in rows:
		token_to_tag_list = list(zip(tokens, tokens_pos))
		output_columns.append((added_so_far, sentence, token_to_tag_list))
		output_columns.append((added_so_far, "(=CORRECTED=) "+sentence, token_to_tag_list))
		output_columns.append(("", "", ""))
	# write_to_output_file
	create_csv_GT_table(output_path="alternative_format.csv", list_of_tuples=output_columns, list_of_column_names=output_column_names)


def get_dataset_from_files_alternative_2(list_of_GT_paths):
	"""

	:param list_of_GT_paths: list of file paths tp read  from
	:return:
	"""
	rows = []
	for input_path in list_of_GT_paths:
		sentences_info = transform_golden(istring=input_path, start_index=len(rows) + 1)
		rows.extend(sentences_info)
	# prepare outputs
	output_columns = []
	output_column_names = ["sent_id", "tag", "content"]
	for added_so_far, sentence, tokens, tokens_pos in rows:
		token_to_tag_list = list(zip(tokens, tokens_pos))
		output_columns.append((added_so_far, "(=SENTENCE     =)", sentence))
		output_columns.append((added_so_far, "(=ORIGINAL  POS=)", token_to_tag_list))
		output_columns.append((added_so_far, "(=CORRECTED POS=)", token_to_tag_list))
		output_columns.append(("", "", ""))
	# write_to_output_file
	create_csv_GT_table(output_path="alternative_format_2.csv", list_of_tuples=output_columns, list_of_column_names=output_column_names)


def get_dataset_from_files_alternative_3(list_of_GT_paths):
	"""

	:param list_of_GT_paths: list of file paths tp read  from
	:return:
	"""
	rows = []
	for input_path in list_of_GT_paths:
		sentences_info = transform_golden(istring=input_path, start_index=len(rows) + 1)
		rows.extend(sentences_info)
	# prepare outputs
	output_columns = []
	output_column_names = ["sent_id", "tag", "content"]
	for added_so_far, sentence, tokens, tokens_pos in rows:
		token_to_tag_list = list(zip(tokens, tokens_pos))
		translated_pos = [ARTICLE_TO_HUMAN_MAP[pos] if pos in ARTICLE_TO_HUMAN_MAP else (pos + "[UNK]") for pos in tokens_pos]
		token_to_translated_tag_list = list(zip(tokens, translated_pos))
		output_columns.append((added_so_far, "(=SENTENCE     =)", sentence))
		output_columns.append((added_so_far, "(=TRANSLATED  POS=)", token_to_translated_tag_list))
		output_columns.append((added_so_far, "(=ORIGINAL  POS=)", token_to_tag_list))
		output_columns.append((added_so_far, "(=CORRECTED POS=)", token_to_tag_list))
		output_columns.append(("", "", ""))
	# write_to_output_file
	create_csv_GT_table(output_path="alternative_format_3.csv", list_of_tuples=output_columns, list_of_column_names=output_column_names)


def get_dataset_from_files_alternative_3_spaces(list_of_GT_paths):
	"""

	:param list_of_GT_paths: list of file paths tp read  from
	:return:
	"""
	rows = []
	for input_path in list_of_GT_paths:
		sentences_info = transform_golden(istring=input_path, start_index=len(rows) + 1)
		rows.extend(sentences_info)
	# prepare outputs
	output_columns = []
	output_column_names = ["sent_id", "tag", "content"]
	for added_so_far, sentence, tokens, tokens_pos in rows:
		token_to_tag_list = list(zip(tokens, tokens_pos))
		translated_pos = [ARTICLE_TO_HUMAN_MAP[pos] if pos in ARTICLE_TO_HUMAN_MAP else (pos + "[UNK]") for pos in tokens_pos]
		# token_to_translated_tag_list = list(zip(tokens, translated_pos))
		token_to_translated_tag_list = [str(translated_tag) for translated_tag in zip(tokens, translated_pos)]
		tuple_lengths = [len(translated_tag_str) for translated_tag_str in token_to_translated_tag_list]
		output_columns.append((added_so_far, "(=SENTENCE       =)", sentence ))
		output_columns.append((added_so_far, "(=TRANSLATED  POS=)", [token_to_translated_tag_list[i].ljust(tuple_lengths[i]) for i in range(len(token_to_translated_tag_list))] ))
		output_columns.append((added_so_far, "(=ORIGINAL    POS=)", [str(token_to_tag_list[i]).ljust(tuple_lengths[i]) for i in range(len(token_to_tag_list))] ))
		output_columns.append((added_so_far, "(=CORRECTED   POS=)", [str(token_to_tag_list[i]).ljust(tuple_lengths[i]) for i in range(len(token_to_tag_list))] ))
		output_columns.append(("", "", ""))
	# write_to_output_file
	create_csv_GT_table(output_path="alternative_format_3_spaces.csv", list_of_tuples=output_columns, list_of_column_names=output_column_names)


def get_dataset_from_files_alternative_3_spaces_2(list_of_GT_paths):
	"""

	:param list_of_GT_paths: list of file paths tp read  from
	:return:
	"""
	rows = []
	for input_path in list_of_GT_paths:
		sentences_info = transform_golden(istring=input_path, start_index=len(rows) + 1)
		rows.extend(sentences_info)
	# prepare outputs
	output_columns = []
	output_column_names = ["sent_id", "tag", "content"]
	for added_so_far, sentence, tokens, tokens_pos in rows:
		token_to_tag_list = list(zip(tokens, tokens_pos))
		translated_pos = [ARTICLE_TO_HUMAN_MAP[pos] if pos in ARTICLE_TO_HUMAN_MAP else (pos + "[UNK]") for pos in tokens_pos]
		# token_to_translated_tag_list = list(zip(tokens, translated_pos))
		token_to_translated_tag_list = [translated_tag for translated_tag in zip(tokens, translated_pos)]
		tuple_lengths = [len(translated_tag[1]) for translated_tag in token_to_translated_tag_list]
		output_columns.append((added_so_far, "(=SENTENCE       =)", sentence ))
		output_columns.append((added_so_far, "(=TRANSLATED  POS=)", [(token_to_translated_tag_list[i][0], token_to_translated_tag_list[i][1].ljust(tuple_lengths[i])) for i in range(len(token_to_translated_tag_list))] ))
		output_columns.append((added_so_far, "(=ORIGINAL    POS=)", [(token_to_tag_list[i][0], token_to_tag_list[i][1].ljust(tuple_lengths[i])) for i in range(len(token_to_tag_list))] ))
		output_columns.append((added_so_far, "(=CORRECTED   POS=)", [(token_to_tag_list[i][0], token_to_tag_list[i][1].ljust(tuple_lengths[i])) for i in range(len(token_to_tag_list))] ))
		output_columns.append(("", "", ""))
	# write_to_output_file
	create_csv_GT_table(output_path="alternative_format_3_spaces_2.csv", list_of_tuples=output_columns, list_of_column_names=output_column_names)


if __name__ == "__main__":
	file_paths = [r"C:\Users\admin\Desktop\newly_annotated_data\golden-cy.txt",
				  r"C:\Users\admin\Desktop\newly_annotated_data\golden-lf.txt",
				  r"C:\Users\admin\Desktop\newly_annotated_data\golden-lj.txt",
				  r"C:\Users\admin\Desktop\newly_annotated_data\golden-ly.txt",
				  r"C:\Users\admin\Desktop\newly_annotated_data\golden-wt.txt",
				  r"C:\Users\admin\Desktop\newly_annotated_data\golden-zc.txt"
				  ]
	# get_dataset_from_files(file_paths)
	# get_dataset_from_files_alternative(file_paths)
	# get_dataset_from_files_alternative_2(file_paths)
	# get_dataset_from_files_alternative_3(file_paths)
	# get_dataset_from_files_alternative_3_spaces(file_paths)
	get_dataset_from_files_alternative_3_spaces_2(file_paths)

